## 20-03-2021
### Applying n-1 transformations for each pointcloud
- Inverting transformation matrix every time.
- Avoiding multiplying with zero each time.
- Finding pointclouds by splitting a bigger array.

|task|time_spent[s]|
|----|-------------|
|apply_transformations 10 pointclouds| 14.219976425170898|
|apply_transformations 9 pointclouds| 13.5791916847229|
|apply_transformations 8 pointclouds| 11.460342168807983|
|apply_transformations 7 pointclouds| 10.091919898986816|
|apply_transformations 6 pointclouds| 8.547046661376953|
|apply_transformations 5 pointclouds| 7.150447845458984|
|apply_transformations 4 pointclouds| 5.737638235092163|
|apply_transformations 3 pointclouds| 4.281572341918945|
|apply_transformations 2 pointclouds| 2.852299690246582|
|apply_transformations 1 pointcloud | 1.4153335094451904|
|||
|**Total time spent**| **79.34012293815613**|
| | | 
|get_vertices_and_texture_from_pointcloud| 2.6218369007110596|

**get_vertices_and_texture_from_pointcloud** takes too much time and it is not suited to be used while getting the frames.

## 20-03-2021 22:30
### Applying n-1 transformations for each pointcloud
- Inverting transformation matrix when computing it then storing it inverted in the list.
- Avoiding multiplying with zero each time.
- Indexing lists to get pointclouds.
- Keeping color frames and texture coordinates in memory in order to get RGB information for each point.
- Saving pointclouds to file after updates.

|task|time_spent[s]|
|----|-------------|
|apply_transformations 10 pointclouds| 7.1543731689453125|
|apply_transformations 9 pointclouds| 6.339238166809082|
|apply_transformations 8 pointclouds| 5.847517251968384|
|apply_transformations 7 pointclouds| 5.086743116378784|
|apply_transformations 6 pointclouds| 4.435328483581543|
|apply_transformations 5 pointclouds| 3.784731864929199|
|apply_transformations 4 pointclouds| 2.964580535888672|
|apply_transformations 3 pointclouds| 2.179731845855713|
|apply_transformations 2 pointclouds| 1.4859611988067627|
|apply_transformations 1 pointcloud | 0.7237105369567871|
|||
|**Total time spent**| **40.00211000442505**|
| | | 
|get_texture_from_pointcloud| 2.068113327026367|


## 02-04-2021 16:00
### Applying n-1 transformations for each pointcloud
- Using the direct transformation matrix.
- Storing only valid points from each pointcloud from the beginning.
- Indexing lists to get pointclouds.
- Keeping color frames and texture coordinates in memory in order to get RGB information for each point.
- Saving pointclouds to file after updates.
- **Using for iteration to multiply each point with the transformation matrix.**

|task|time_spent[s]|
|----|-------------|
|apply_transformations 1 pointcloud | 0.24056720733642578|
|||
|**Total time spent for 10 pointclouds**| **12.109205722808838**|
| | | 
|get_texture_from_pointcloud| 1.696603775024414|


## Comparison between methods used to multiply a pointcloud with the transformation matrix

|task|method|time_spent[s]|**Total time spent for 10 pointclouds**|
|----|-------------|-------|--------------|
|apply_transformations 1 pointcloud | Multiply using **for** | 0.24056720733642578 | 12.109205722808838|
|apply_transformations 1 pointcloud | using **np.einsum**| 0.006560802459716797 | 0.32988429069519043 |
|apply_transformations 1 pointcloud | using **.dot**| 0.003931999206542969 | 0.20569825172424316 |

