import numpy as np


def asvoid(arr):
    """
    Based on http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
    View the array as dtype np.void (bytes). The items along the last axis are
    viewed as one value. This allows comparisons to be performed which treat
    entire rows as one value.
    """
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        """Care needs to be taken here since
        np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
        Adding 0. converts -0. to 0.
        """
        arr += 0.0
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


arr = [
    [True, False, True],
    [True, False, True],
    [False, True, True],
    [True, True, False],
    [True, False, True],
]

search_arr = [
    [True, False, True],
    [False, True, True],
    [True, True, False],
]

# arr = asvoid(np.array(arr))
# search_arr = asvoid(np.array(search_arr)[0, :])

arr = np.array(arr)
search_arr = np.array(search_arr)
print((arr == search_arr[0, :]).all(axis=1))
# idx = np.flatnonzero(np.in1d(asvoid(arr), asvoid(search_arr)))
# print(idx)
