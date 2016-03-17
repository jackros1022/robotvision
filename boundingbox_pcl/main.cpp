#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
// typedef pcl::PointXYZRGBA Point_Type;
typedef pcl::PointXYZ Point_Type;

boost::shared_ptr<pcl::visualization::PCLVisualizer>
simpleVis (pcl::PointCloud<Point_Type>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<Point_Type> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  //viewer->addCoordinateSystem (1.0, "global");
  viewer->initCameraParameters ();
  return (viewer);
}

int
main (int argc, char** argv)
{
    if( argc != 3){
        std::cout<< "Input format is wrong" << std::endl;
        return (0);
    }

     // Read in the cloud data
    pcl::PCDReader reader;
    pcl::PointCloud<Point_Type>::Ptr cloud (new pcl::PointCloud<Point_Type>), cloud_f (new pcl::PointCloud<Point_Type>);
    reader.read (argv[1], *cloud);
    std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*
    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<Point_Type> vg;
    pcl::PointCloud<Point_Type>::Ptr cloud_filtered (new pcl::PointCloud<Point_Type>);
    vg.setInputCloud (cloud);
    vg.setLeafSize (0.01f, 0.01f, 0.01f);
    vg.filter (*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*
    
    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<Point_Type> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<Point_Type>::Ptr cloud_plane (new pcl::PointCloud<Point_Type> ());
    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.02);
    
    int i=0, nr_points = (int) cloud_filtered->points.size ();
    while (cloud_filtered->points.size () > 0.3 * nr_points)
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }
        
        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<Point_Type> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);
        
        // Get the points associated with the planar surface
        extract.filter (*cloud_plane);
        std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
        
        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_f);
        *cloud_filtered = *cloud_f;
    }
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<Point_Type>::Ptr tree (new pcl::search::KdTree<Point_Type>);
    tree->setInputCloud (cloud_filtered);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<Point_Type> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);
        
    if(pcl::console::find_argument (argc, argv, "-1") >= 0){
        std::cout<< "mode 1" << std::endl;
    
        
        int j = 0;
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {
           
            pcl::PointCloud<Point_Type>::Ptr cloud_cluster (new pcl::PointCloud<Point_Type>);
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
                cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
            cloud_cluster->width = cloud_cluster->points.size ();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            
            std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
            std::stringstream ss;
            ss << "cloud_cluster_" << j << ".pcd";
            writer.write<Point_Type> (ss.str (), *cloud_cluster, false); //*
            j++;
        }
         std::cout<<"number of objects is "<< j<< "" <<std::endl;
        
        return (0);
    }
    else if (pcl::console::find_argument (argc, argv, "-2") >= 0 ){
        std::cout<< "mode 2" << std::endl;

        std::cout<< "Default mode, print the 3D image" << std::endl;
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
        viewer = simpleVis(cloud);

  pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
  feature_extractor.setInputCloud (cloud);
  feature_extractor.compute ();

  std::vector <float> moment_of_inertia;
  std::vector <float> eccentricity;
  pcl::PointXYZ min_point_AABB;
  pcl::PointXYZ max_point_AABB;
  pcl::PointXYZ min_point_OBB;
  pcl::PointXYZ max_point_OBB;
  pcl::PointXYZ position_OBB;
  Eigen::Matrix3f rotational_matrix_OBB;
  float major_value, middle_value, minor_value;
  Eigen::Vector3f major_vector, middle_vector, minor_vector;
  Eigen::Vector3f mass_center;

  feature_extractor.getMomentOfInertia (moment_of_inertia);
  feature_extractor.getEccentricity (eccentricity);
  feature_extractor.getAABB (min_point_AABB, max_point_AABB);
  feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
  feature_extractor.getEigenValues (major_value, middle_value, minor_value);
  feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
  feature_extractor.getMassCenter (mass_center);

  viewer->addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB");

  Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);
  Eigen::Quaternionf quat (rotational_matrix_OBB);
  viewer->addCube (position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");

  pcl::PointXYZ center (mass_center (0), mass_center (1), mass_center (2));
  pcl::PointXYZ x_axis (major_vector (0) + mass_center (0), major_vector (1) + mass_center (1), major_vector (2) + mass_center (2));
  pcl::PointXYZ y_axis (middle_vector (0) + mass_center (0), middle_vector (1) + mass_center (1), middle_vector (2) + mass_center (2));
  pcl::PointXYZ z_axis (minor_vector (0) + mass_center (0), minor_vector (1) + mass_center (1), minor_vector (2) + mass_center (2));
  viewer->addLine (center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector");
  viewer->addLine (center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector");
  viewer->addLine (center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");
       
        int j = 0;
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {
           
            pcl::PointCloud<Point_Type>::Ptr cloud_cluster (new pcl::PointCloud<Point_Type>);
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
                cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
            cloud_cluster->width = cloud_cluster->points.size ();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            
            std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
            std::stringstream ss;
            ss << "cloud_cluster_" << j << ".pcd";
            writer.write<Point_Type> (ss.str (), *cloud_cluster, false); //*
            j++;
            pcl::PointXYZ minPt, maxPt;
            pcl::getAABB(*cloud_cluster, minPt, maxPt);
            std::cout << "Max x: " << maxPt.x << std::endl;
            std::cout << "Max y: " << maxPt.y << std::endl;
            std::cout << "Max z: " << maxPt.z << std::endl;
            std::cout << "Min x: " << minPt.x << std::endl;
            std::cout << "Min y: " << minPt.y << std::endl;
            std::cout << "Min z: " << minPt.z << std::endl;
            viewer->addCube (minPt.x, maxPt.x, minPt.y, maxPt.y, minPt.z, maxPt.z, 1.0, 1.0, 0.5, ss.str());

        }
         std::cout<<"number of objects is "<< j<< "" <<std::endl;

        while (!viewer->wasStopped ()){
            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
      return 0;
        

        
    }
    else if (pcl::console::find_argument (argc, argv, "-3") >= 0 ){
        std::cout<< "mode 3" << std::endl;
    }
    else {
        std::cout<< "Default mode, print the 3D image" << std::endl;
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
        viewer = simpleVis(cloud);
        while (!viewer->wasStopped ()){
            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
      return 0;
    }

}