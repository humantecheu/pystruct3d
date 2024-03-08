import numpy as np


class Munkres:
    def __init__(self) -> None:
        self.cost_matrix = None
        self.mask_matrix = None
        self.n = 0

        self.covered_rows = None
        self.covered_cols = None

        self.uncovered_prime = None

    def compute(self, cost_matrix: np.ndarray, maximize: bool = False):
        max_value = np.max(cost_matrix)
        if maximize:
            step, n, m = self.__step_zero(max_value - cost_matrix)
        else:
            step, n, m = self.__step_zero(cost_matrix)

        while step is not None:
            step = step()

        rows, cols = np.where(self.mask_matrix[:n, :m] == 1)
        return np.vstack((rows, cols)).T

    def __step_zero(self, cost_matrix: np.ndarray) -> callable:
        n, m = cost_matrix.shape
        if n == m:  # Already square cost_matrix
            max_size = n
            padded_matrix = cost_matrix.copy()
        else:
            max_size = max(n, m)  # Find the larger dimension
            padded_matrix = np.zeros((max_size, max_size))  # Create a square matrix
            # Copy the original matrix into the top-left corner
            padded_matrix[:n, :m] = cost_matrix.copy()

        self.cost_matrix = padded_matrix
        self.mask_matrix = np.zeros_like(self.cost_matrix)
        self.n = max_size

        self.covered_rows = np.zeros((self.n,))
        self.covered_cols = np.zeros((self.n,))

        return self.__step_one, n, m

    def __step_one(self) -> callable:
        row_min = np.min(self.cost_matrix, axis=1)[np.newaxis, :].T
        self.cost_matrix -= row_min

        return self.__step_two

    def __step_two(self) -> callable:
        zeros = np.where(self.cost_matrix == 0)

        for r, c in zip(zeros[0], zeros[1]):
            if self.covered_rows[r] or self.covered_cols[c]:
                continue
            self.mask_matrix[r, c] = 1
            self.covered_rows[r] = 1
            self.covered_cols[c] = 1

        self.covered_rows *= 0
        self.covered_cols *= 0

        return self.__step_three

    def __step_three(self) -> int:
        starred = np.where(self.mask_matrix == 1)
        self.covered_cols[np.unique(starred[1])] = 1

        if np.sum(self.covered_cols) >= self.n:
            return None  # Done
        else:
            return self.__step_four  # Go to step 4

    def __step_four(self) -> callable:
        zeros = self.cost_matrix == 0
        zeros[self.covered_rows == 1, :] = False
        zeros[:, self.covered_cols == 1] = False
        uncovered_zero = np.where(zeros)

        if uncovered_zero[0].size:
            r, c = uncovered_zero[0][0], uncovered_zero[1][0]
            self.mask_matrix[r, c] = 2
            starred_in_row = np.where(self.mask_matrix[r, :] == 1)
            if starred_in_row[0].size:
                c = starred_in_row[0][0]
                self.covered_rows[r] = 1
                self.covered_cols[c] = 0
                return self.__step_four  # Repeat until no uncovered zeros left
            else:
                self.uncovered_prime = (r, c)
                return self.__step_five
        else:
            return self.__step_six

    def __step_five(self) -> callable:
        path = [self.uncovered_prime]
        while True:
            c = path[-1][1]
            starred_in_col = np.where(self.mask_matrix[:, c] == 1)
            if starred_in_col[0].size:
                r = starred_in_col[0][0]
                path.append((r, c))
            else:
                break

            r = path[-1][0]
            primed_in_row = np.where(self.mask_matrix[r, :] == 2)
            c = primed_in_row[0][0]  # if primed_in_row[0].size else path[-1][1]
            path.append((r, c))

        # Augment path into mask_matrix
        path = np.array(path)
        self.mask_matrix[path[:, 0], path[:, 1]] = np.where(
            self.mask_matrix[path[:, 0], path[:, 1]] == 1, 0, 1
        )
        # Clear primed zeros
        self.mask_matrix = np.where(self.mask_matrix == 2, 0, self.mask_matrix)
        self.covered_rows *= 0
        self.covered_cols *= 0
        return self.__step_three

    def __step_six(self) -> callable:
        # Get smallest uncovered value in cost_matrix
        mask = np.outer(self.covered_rows == 0, self.covered_cols == 0)
        uncovered_cost = self.cost_matrix[mask]
        min_uncovered = np.min(uncovered_cost)

        self.cost_matrix[self.covered_rows == 1, :] += min_uncovered
        self.cost_matrix[:, self.covered_cols == 0] -= min_uncovered

        return self.__step_four


def main():
    munkres = Munkres()
    cost_matrix = np.array(
        [
            [7, 6, 2, 9, 2, 6],
            [6, 2, 1, 3, 9, 1],
            [5, 6, 8, 9, 5, 4],
            [6, 8, 5, 8, 6, 5],
            [9, 5, 6, 4, 7, 9],
        ]
    )

    res = munkres.compute(cost_matrix)
    print(res)
    print(np.sum(cost_matrix[res[:, 0], res[:, 1]]))


if __name__ == "__main__":
    main()
