import numpy as np

import torch

PI = 3.14159


def encode_label(K, ry, dims, locs):
    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]

    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]

    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2

    corners_3d = np.array([x_corners, y_corners, z_corners])
    rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = np.matmul(rot_mat, corners_3d)
    corners_3d += np.array([x, y, z]).reshape([3, 1])

    loc_center = np.array([x, y - h / 2, z])
    proj_point = np.matmul(K, loc_center)
    proj_point = proj_point[:2] / proj_point[2]

    corners_2d = np.matmul(K, corners_3d)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([min(corners_2d[0]), min(corners_2d[1]),
                      max(corners_2d[0]), max(corners_2d[1])])

    return proj_point, box2d, corners_3d


class SMOKECoder():
    def __init__(self, depth_ref, dim_ref, device="cuda"):
        self.depth_ref = torch.as_tensor(depth_ref).to(device=device)
        self.dim_ref = torch.as_tensor(dim_ref).to(device=device)

    def encode_box2d(self, K, rotys, dims, locs, img_size):
        device = rotys.device
        K = K.to(device=device)

        img_size = img_size.flatten()

        box3d = self.encode_box3d(rotys, dims, locs)
        box3d_image = torch.matmul(K, box3d)
        box3d_image = box3d_image[:, :2, :] / box3d_image[:, 2, :].view(
            box3d.shape[0], 1, box3d.shape[2]
        )

        xmins, _ = box3d_image[:, 0, :].min(dim=1)
        xmaxs, _ = box3d_image[:, 0, :].max(dim=1)
        ymins, _ = box3d_image[:, 1, :].min(dim=1)
        ymaxs, _ = box3d_image[:, 1, :].max(dim=1)

        xmins = xmins.clamp(0, img_size[0])
        xmaxs = xmaxs.clamp(0, img_size[0])
        ymins = ymins.clamp(0, img_size[1])
        ymaxs = ymaxs.clamp(0, img_size[1])

        bboxfrom3d = torch.cat((xmins.unsqueeze(1), ymins.unsqueeze(1),
                                xmaxs.unsqueeze(1), ymaxs.unsqueeze(1)), dim=1)

        return bboxfrom3d

    @staticmethod
    def rad_to_matrix(rotys, N):
        device = rotys.device

        cos, sin = rotys.cos(), rotys.sin()

        i_temp = torch.tensor([[1, 0, 1],
                               [0, 1, 0],
                               [-1, 0, 1]]).to(dtype=torch.float32,
                                               device=device)
        ry = i_temp.repeat(N, 1).view(N, -1, 3)

        ry[:, 0, 0] *= cos
        ry[:, 0, 2] *= sin
        ry[:, 2, 0] *= sin
        ry[:, 2, 2] *= cos

        return ry

    def encode_box3d(self, rotys, dims, locs):
        '''
        construct 3d bounding box for each object.
        Args:
            rotys: rotation in shape N
            dims: dimensions of objects
            locs: locations of objects

        Returns:

        '''
        if len(rotys.shape) == 2:
            rotys = rotys.flatten()
        if len(dims.shape) == 3:
            dims = dims.view(-1, 3)
        if len(locs.shape) == 3:
            locs = locs.view(-1, 3)

        device = rotys.device
        N = rotys.shape[0]
        ry = self.rad_to_matrix(rotys, N)

        dims = dims.view(-1, 1).repeat(1, 8)
        dims[::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[2::3, :4]
        dims[::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[2::3, 4:]
        dims[1::3, :4], dims[1::3, 4:] = 0., -dims[1::3, 4:]
        index = torch.tensor([[4, 0, 1, 2, 3, 5, 6, 7],
                              [4, 5, 0, 1, 6, 7, 2, 3],
                              [4, 5, 6, 0, 1, 2, 3, 7]]).repeat(N, 1).to(device=device)
        box_3d_object = torch.gather(dims, 1, index)
        box_3d = torch.matmul(ry, box_3d_object.view(N, 3, -1))
        box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

        return box_3d

    def decode_depth(self, depths_offset):
        '''
        Transform depth offset to depth
        '''
        depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]

        return depth

    def decode_location(self,
                        points,
                        points_offset,
                        depths,
                        Ks,
                        trans_mats):
        '''
        retrieve objects location in camera coordinate based on projected points
        Args:
            points: projected points on feature map in (x, y)
            points_offset: project points offset in (delata_x, delta_y)
            depths: object depth z
            Ks: camera intrinsic matrix, shape = [N, 3, 3]
            trans_mats: transformation matrix from image to feature map, shape = [N, 3, 3]

        Returns:
            locations: objects location, shape = [N, 3]
        '''
        device = points.device

        Ks = Ks.to(device=device)
        trans_mats = trans_mats.to(device=device)

        # number of points
        N = points_offset.shape[0]
        # batch size
        N_batch = Ks.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()

        trans_mats_inv = trans_mats.inverse()[obj_id]
        Ks_inv = Ks.inverse()[obj_id]

        points = points.view(-1, 2)
        assert points.shape[0] == N
        proj_points = points + points_offset
        # transform project points in homogeneous form.
        proj_points_extend = torch.cat(
            (proj_points, torch.ones(N, 1).to(device=device)), dim=1)
        # expand project points as [N, 3, 1]
        proj_points_extend = proj_points_extend.unsqueeze(-1)
        # transform project points back on image
        proj_points_img = torch.matmul(trans_mats_inv, proj_points_extend)
        # with depth
        proj_points_img = proj_points_img * depths.view(N, -1, 1)
        # transform image coordinates back to object locations
        locations = torch.matmul(Ks_inv, proj_points_img)

        return locations.squeeze(2)

    def decode_dimension(self, cls_id, dims_offset):
        '''
        retrieve object dimensions
        Args:
            cls_id: each object id
            dims_offset: dimension offsets, shape = (N, 3)

        Returns:

        '''
        cls_id = cls_id.flatten().long()

        dims_select = self.dim_ref[cls_id, :]
        dimensions = dims_offset.exp() * dims_select

        return dimensions

    def decode_orientation(self, vector_ori, locations, flip_mask=None):
        '''
        retrieve object orientation
        Args:
            vector_ori: local orientation in [sin, cos] format
            locations: object location

        Returns: for training we only need roty
                 for testing we need both alpha and roty

        '''

        locations = locations.view(-1, 3)
        rays = torch.atan(locations[:, 0] / (locations[:, 2] + 1e-7))
        alphas = torch.atan(vector_ori[:, 0] / (vector_ori[:, 1] + 1e-7))

        # get cosine value positive and negtive index.
        cos_pos_idx = (vector_ori[:, 1] >= 0).nonzero()
        cos_neg_idx = (vector_ori[:, 1] < 0).nonzero()

        alphas[cos_pos_idx] -= PI / 2
        alphas[cos_neg_idx] += PI / 2

        # retrieve object rotation y angle.
        rotys = alphas + rays

        # in training time, it does not matter if angle lies in [-PI, PI]
        # it matters at inference time? todo: does it really matter if it exceeds.
        larger_idx = (rotys > PI).nonzero()
        small_idx = (rotys < -PI).nonzero()

        if len(larger_idx) != 0:
            rotys[larger_idx] -= 2 * PI
        if len(small_idx) != 0:
            rotys[small_idx] += 2 * PI

        if flip_mask is not None:
            fm = flip_mask.flatten()
            rotys_flip = fm.float() * rotys

            rotys_flip_pos_idx = rotys_flip > 0
            rotys_flip_neg_idx = rotys_flip < 0
            rotys_flip[rotys_flip_pos_idx] -= PI
            rotys_flip[rotys_flip_neg_idx] += PI

            rotys_all = fm.float() * rotys_flip + (1 - fm.float()) * rotys

            return rotys_all

        else:
            return rotys, alphas


if __name__ == '__main__':
    sc = SMOKECoder(depth_ref=(28.01, 16.32),
                    dim_ref=((3.88, 1.63, 1.53),
                             (1.78, 1.70, 0.58),
                             (0.88, 1.73, 0.67)))
    depth_offset = torch.tensor([-1.3977,
                                 -0.9933,
                                 0.0000,
                                 -0.7053,
                                 -0.3652,
                                 -0.2678,
                                 0.0650,
                                 0.0319,
                                 0.9093])
    depth = sc.decode_depth(depth_offset)
    print(depth)

    points = torch.tensor([[4, 75],
                           [200, 59],
                           [0, 0],
                           [97, 54],
                           [105, 51],
                           [165, 52],
                           [158, 50],
                           [111, 50],
                           [143, 48]], )
    points_offset = torch.tensor([[0.5722, 0.1508],
                                  [0.6010, 0.1145],
                                  [0.0000, 0.0000],
                                  [0.0365, 0.1977],
                                  [0.0267, 0.7722],
                                  [0.9360, 0.0118],
                                  [0.8549, 0.5895],
                                  [0.6011, 0.6448],
                                  [0.4246, 0.4782]], )
    K = torch.tensor([[721.54, 0., 631.44],
                      [0., 721.54, 172.85],
                      [0, 0, 1]]).unsqueeze(0)

    trans_mat = torch.tensor([[2.5765e-01, -0.0000e+00, 2.5765e-01],
                              [-2.2884e-17, 2.5765e-01, -3.0918e-01],
                              [0, 0, 1]], ).unsqueeze(0)
    locations = sc.decode_location(points, points_offset, depth, K, trans_mat)

    cls_ids = torch.tensor([[0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0]])

    dim_offsets = torch.tensor([[-0.0375, 0.0755, -0.1469],
                                [-0.1309, 0.1054, 0.0179],
                                [0.0000, 0.0000, 0.0000],
                                [-0.0765, 0.0447, -0.1803],
                                [-0.1170, 0.1286, 0.0552],
                                [-0.0568, 0.0935, -0.0235],
                                [-0.0898, -0.0066, -0.1469],
                                [-0.0633, 0.0755, 0.1189],
                                [0.0061, -0.0537, -0.1088]]).roll(1, 1)
    dimensions = sc.decode_dimension(cls_ids, dim_offsets)

    locations[:, 1] += dimensions[:, 1] / 2
    print(locations)
    print(dimensions)

    vector_ori = torch.tensor([[0.4962, 0.8682],
                               [0.3702, -0.9290],
                               [0.0000, 0.0000],
                               [0.2077, 0.9782],
                               [0.1189, 0.9929],
                               [0.2272, -0.9738],
                               [0.1979, -0.9802],
                               [0.0990, 0.9951],
                               [0.3421, -0.9396]])
    flip_mask = torch.tensor([1, 1, 0, 1, 1, 1, 1, 1, 1])
    rotys = sc.decode_orientation(vector_ori, locations, flip_mask)
    print(rotys)
    rotys = torch.tensor([[1.4200],
                          [-1.7600],
                          [0.0000],
                          [1.4400],
                          [1.3900],
                          [-1.7800],
                          [-1.7900],
                          [1.4000],
                          [-2.0200]])
    box3d = sc.encode_box3d(rotys, dimensions, locations)
    print(box3d)
