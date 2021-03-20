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