import numpy as np
from scipy import spatial


class PointCloud:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        self.nx = None
        self.ny = None
        self.nz = None
        self.planarity = None

        self.no_points = len(x)
        self.sel = None

    def select_n_points(self, n):
        if self.no_points > n:
            self.sel = np.round(np.linspace(0, self.no_points - 1, n)).astype(int)
        else:
            self.sel = np.arange(0, self.no_points, 1)

    def estimate_normals(self, neighbors):

        self.nx = np.full(self.no_points, np.nan)
        self.ny = np.full(self.no_points, np.nan)
        self.nz = np.full(self.no_points, np.nan)
        self.planarity = np.full(self.no_points, np.nan)

        kdtree = spatial.cKDTree(np.column_stack((self.x, self.y, self.z)))
        query_points = np.column_stack((self.x[self.sel], self.y[self.sel], self.z[self.sel]))
        _, idxNN_all_qp = kdtree.query(query_points, k=neighbors, p=2, n_jobs=-1)

        for (i, idxNN) in enumerate(idxNN_all_qp):
            selected_points = np.column_stack((self.x[idxNN], self.y[idxNN], self.z[idxNN]))
            C = np.cov(selected_points.T, bias=False)
            eig_vals, eig_vecs = np.linalg.eig(C)
            idx_sort = eig_vals.argsort()[::-1]  # sort from large to small
            eig_vals = eig_vals[idx_sort]
            eig_vecs = eig_vecs[:, idx_sort]
            self.nx[self.sel[i]] = eig_vecs[0, 2]
            self.ny[self.sel[i]] = eig_vecs[1, 2]
            self.nz[self.sel[i]] = eig_vecs[2, 2]
            self.planarity[self.sel[i]] = (eig_vals[1] - eig_vals[2]) / eig_vals[0]

    def transform(self, H):
        """
        Apply transformation matrix H to current point_cloud
        and update it's values.
        """

        euler_X = np.column_stack((self.x, self.y, self.z))
        homogeneous_X = PointCloud.euler_coord_to_homogeneous_coord(euler_X)
        out_homogeneous_X = np.transpose(H @ homogeneous_X.T)
        out_X = PointCloud.homogeneous_coord_to_euler_coord(out_homogeneous_X)

        self.x = out_X[:, 0]
        self.y = out_X[:, 1]
        self.z = out_X[:, 2]

    @staticmethod
    def euler_coord_to_homogeneous_coord(euler_X):
        """
        Append a column at the end of point cloud to transform it
        from euler coordinates to homogeneous coordinates.
        """

        no_points = np.shape(euler_X)[0]
        homogeneous_X = np.column_stack((euler_X, np.ones(no_points)))

        return homogeneous_X

    @staticmethod
    def homogeneous_coord_to_euler_coord(homogeneous_X):
        """
        Transform from homogeneous coordinates to euler coordinates
        by dividing each column with the last column value.
        """

        euler_X = np.column_stack((
            homogeneous_X[:, 0] / homogeneous_X[:, 3],
            homogeneous_X[:, 1] / homogeneous_X[:, 3],
            homogeneous_X[:, 2] / homogeneous_X[:, 3]
        ))

        return euler_X
