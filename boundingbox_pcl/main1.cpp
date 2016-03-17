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
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>


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

#include <pcl/features/moment_of_inertia_estimation.h>
#include <vector>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/pca.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>

#include <pcl/console/print.h>
#include <boost/filesystem.hpp>
#include <pcl/common/io.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;
using namespace pcl;
// --------------
// -----Help-----
// --------------
void
printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" [options]\n\n"
            << "Options:\n"
            << "-------------------------------------------\n"
            << "-h           this help\n"
            << "-s           Simple visualisation example\n"
            << "-r           RGB colour visualisation example\n"
            << "\n\n";
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}



unsigned int text_id = 0;
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
  if (event.getKeySym () == "r" && event.keyDown ())
  {
    std::cout << "r was pressed => removing all text" << std::endl;

    char str[512];
    for (unsigned int i = 0; i < text_id; ++i)
    {
      sprintf (str, "text#%03d", i);
      viewer->removeShape (str);
    }
    text_id = 0;
  }
}

void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
                         void* viewer_void)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
  if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
      event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
  {
    std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;

    char str[512];
    sprintf (str, "text#%03d", text_id ++);
    viewer->addText ("clicked here", event.getX (), event.getY (), str);
  }
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> interactionCustomizationVis ()
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1.0);

  viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
  viewer->registerMouseCallback (mouseEventOccurred, (void*)&viewer);

  return (viewer);
}


// --------------
// -----Main-----
// --------------
int
main (int argc, char** argv)
{
  // --------------------------------------
  // -----Parse Command Line Arguments-----
  // --------------------------------------
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    printUsage (argv[0]);
    return 0;
  }
  bool simple(false), rgb(false), custom_c(false), normals(false),
    shapes(false), viewports(false), interaction_customization(false);
  if (pcl::console::find_argument (argc, argv, "-s") >= 0)
  {
    simple = true;
    std::cout << "Simple visualisation example\n";
  }
  else if (pcl::console::find_argument (argc, argv, "-r") >= 0)
  {
    rgb = true;
    std::cout << "RGB colour visualisation example\n";
  }
  else
  {
    printUsage (argv[0]);
    return 0;
  }


   

  // Read in the cloud data
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr final (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>), cloud_f (new pcl::PointCloud<pcl::PointXYZRGB>);
    reader.read (argv[1], *basic_cloud_ptr);
    reader.read (argv[1], *cloud_ptr);
    std::cout << "PointCloud before filtering has: " << cloud_ptr->points.size () << " data points." << std::endl; //*

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    vg.setInputCloud (cloud_ptr);
    vg.setLeafSize (0.01f, 0.01f, 0.01f);
    vg.filter (*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*


  


    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::PointIndices::Ptr inliers2 (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::PCDWriter writer;
    
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.01);
    
    int i=0, nr_points = (int) cloud_filtered->points.size ();
    // while (cloud_filtered->points.size () > 0.8 * nr_points)
    // {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        // if (inliers->indices.size () == 0)
        // {
        //     std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
        //     break;
        // }
        
        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
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

        seg.setInputCloud (cloud_plane);
        seg.segment (*inliers2, *coefficients);
    // }
        // print out the plance coefficients
         std::cout << "Model coefficients: " 
          << coefficients->values[0] << " " 
          << coefficients->values[1] << " "
          << coefficients->values[2] << " " 
          << coefficients->values[3] << std::endl;

        // pcl::PointXYZ normal (coefficients->values[0],  coefficients->values[1],  coefficients->values[2]);
        // viewer->addLine (cloud_filtered->points[100], normal , 1.0f, 1.0f, 1.0f, "normal");
    // copies all inliers of the model computed to another PointCloud
    //pcl::copyPointCloud<pcl::PointXYZ>(*cloud_filtered, inliers, *the_plane);


    //find the mass center of cloud
    pcl::MomentOfInertiaEstimation <pcl::PointXYZRGB> feature_extractor; 
    Eigen::Vector3f mass_center;
    feature_extractor.setInputCloud (cloud_filtered);
    feature_extractor.compute ();
    feature_extractor.getMassCenter (mass_center);

    // Rotate the cloud due to the plane coefficient
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_rotated (new pcl::PointCloud<pcl::PointXYZRGB>);
    Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
    Eigen::Matrix<float, 1, 3> fitted_plane_norm, xyaxis_plane_norm, rotation_axis;
    fitted_plane_norm[0] = coefficients->values[0];
    fitted_plane_norm[1] = coefficients->values[1];
    fitted_plane_norm[2] = coefficients->values[2];
    xyaxis_plane_norm[0] = 0.0;
    xyaxis_plane_norm[1] = 0.0;
    xyaxis_plane_norm[2] = 1.0;
    rotation_axis = xyaxis_plane_norm.cross(fitted_plane_norm);
    float theta = -acos(xyaxis_plane_norm.dot(fitted_plane_norm)); 
    //float theta = -atan2(rotation_axis.norm(), xyaxis_plane_norm.dot(fitted_plane_norm));
    transform_2.rotate(Eigen::AngleAxisf(theta, rotation_axis.normalized()));
    transform_2.translation()<< mass_center[0], mass_center[1] - 0.8, mass_center[2];

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_recentered(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_recentered2(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*cloud_filtered, *cloud_recentered, transform_2);
    pcl::transformPointCloud(*cloud_ptr, *cloud_recentered2, transform_2);
    


    // open the viewer, show the cloud
     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    if (simple)
    {
      viewer = simpleVis(basic_cloud_ptr);
    }
    else if (rgb)
    {
      viewer = rgbVis(cloud_recentered2);
    }
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud (cloud_recentered);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance (0.03); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_recentered);
    ec.extract (cluster_indices);



    

  
    int j = 0;
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {

           
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
                cloud_cluster->points.push_back (cloud_recentered->points[*pit]); //*
            cloud_cluster->width = cloud_cluster->points.size ();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            
            std::cout << "\n\n==================The "<< j+1 <<"th Cluster: " << cloud_cluster->points.size () << " data points.======================" << std::endl;
            std::stringstream ss;
            ss << "cloud_cluster_" << j << ".pcd";
            writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_cluster, false); //*
            j++;


            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>); 
            pcl::copyPointCloud(*cloud_cluster, *cloud_xyz);

            ///////////////////////////////////////////////////////////////////
            Eigen::Vector4f pcaCentroid;
            pcl::compute3DCentroid(*cloud_xyz, pcaCentroid);
            Eigen::Matrix3f covariance;
            
            computeCovarianceMatrixNormalized(*cloud_xyz, pcaCentroid, covariance);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
            Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
            // std::cout << eigenVectorsPCA.size() << std::endl;
            Eigen::Matrix<float, 3, 1> z_axis;
            z_axis[0] = 0.0;
            z_axis[1] = 0.0;
            z_axis[2] = 1.0;
            eigenVectorsPCA.col(2) = z_axis;
           
            eigenVectorsPCA.col(1) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(2));
            eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
            eigenVectorsPCA.col(0) = eigenVectorsPCA.col(0).normalized();
            eigenVectorsPCA.col(1) = eigenVectorsPCA.col(1).normalized();

            Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
            projectionTransform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
            projectionTransform.block<3, 1>(0, 3) = -1.f * (projectionTransform.block<3, 3>(0, 0) * pcaCentroid.head<3>());
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*cloud_xyz, *cloudPointsProjected, projectionTransform);
            // Get the minimum and maximum points of the transformed cloud.
            pcl::PointXYZ minPoint, maxPoint;
            pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
            Eigen::Vector3f meanDiagonal = 0.5f*(maxPoint.getVector3fMap() + minPoint.getVector3fMap());
            Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA); 
            Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();
            viewer->addCube(bboxTransform, bboxQuaternion, maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z,"minmax_"+std::to_string(j));
            std::cout << " -----------------before rotate the box----------------------"<< std::endl;
            std::cout << "    Eigenvectors:\n"<< eigenVectorsPCA.transpose() << std::endl;
            std::cout << "    Box length:\n"<< maxPoint.getVector3fMap().transpose() - minPoint.getVector3fMap().transpose()<< std::endl;
            // viewer->addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 0.0, 0.0, "AABB_"+ std::to_string(j));
            // Eigen::Quaternionf quat (rotational_matrix_OBB);
            // viewer->addCube (position, projectionTransform, maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z, "minmax_"+std::to_string(j));
            // viewer->addCube(minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, minPoint.z , maxPoint.z, 1, 1, 0,"minmax_"+std::to_string(j));
            
            //rotate the Eigen vectors in x-y plane , redo the getminmax3D  to search for cylinder.
            Eigen::Matrix3f eigenVectorsPCA2;
            eigenVectorsPCA2.col(2)=eigenVectorsPCA.col(2);
            // eigenVectorsPCA2.col(0)=(maxPoint.x - minPoint.x)*eigenVectorsPCA.col(0)+(maxPoint.y - minPoint.y)*eigenVectorsPCA.col(1);
            // eigenVectorsPCA2.col(1)=(maxPoint.y - minPoint.y)*eigenVectorsPCA.col(0)-(maxPoint.x - minPoint.x)*eigenVectorsPCA.col(1);
            eigenVectorsPCA2.col(0)=eigenVectorsPCA.col(0)+eigenVectorsPCA.col(1);
            eigenVectorsPCA2.col(1)=-eigenVectorsPCA.col(0)+eigenVectorsPCA.col(1);
            eigenVectorsPCA2.col(0)=eigenVectorsPCA2.col(0).normalized();
            eigenVectorsPCA2.col(1)=eigenVectorsPCA2.col(1).normalized();

            Eigen::Matrix4f projectionTransform2(Eigen::Matrix4f::Identity());
            projectionTransform2.block<3, 3>(0, 0) = eigenVectorsPCA2.transpose();
            projectionTransform2.block<3, 1>(0, 3) = -1.f * (projectionTransform2.block<3, 3>(0, 0) * pcaCentroid.head<3>());
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected2(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*cloud_xyz, *cloudPointsProjected2, projectionTransform2);
            // Get the minimum and maximum points of the transformed cloud.
            pcl::PointXYZ minPoint2, maxPoint2;
            pcl::getMinMax3D(*cloudPointsProjected2, minPoint2, maxPoint2);
            Eigen::Vector3f meanDiagonal2 = 0.5f*(maxPoint2.getVector3fMap() + minPoint2.getVector3fMap());
            Eigen::Quaternionf bboxQuaternion2(eigenVectorsPCA2); 
            Eigen::Vector3f bboxTransform2 = eigenVectorsPCA2 * meanDiagonal2 + pcaCentroid.head<3>();
            viewer->addCube(bboxTransform2, bboxQuaternion2, maxPoint2.x - minPoint2.x, maxPoint2.y - minPoint2.y, maxPoint2.z - minPoint2.z,"2minmax_"+std::to_string(j));
            std::cout << " -----------------after rotate the box----------------------"<< std::endl;
            std::cout << "    Eigenvectors:\n"<< eigenVectorsPCA2.transpose() << std::endl;
            std::cout << "    Box length:\n"<< maxPoint2.getVector3fMap().transpose() - minPoint2.getVector3fMap().transpose()<< std::endl;
            //calculate the ratio of box length.  after rotate box length/before rotate box lenth
            float ratio=(maxPoint2.x - minPoint2.x+maxPoint2.y - minPoint2.y)/(maxPoint.x - minPoint.x+maxPoint.y - minPoint.y);

            std::cout << "    Ratio of bounding box after/before rotation = "<< ratio << std::endl;
            if (ratio > 1.1) 
                std::cout << "It is a box!  length="<< maxPoint.y - minPoint.y <<" width=" << maxPoint.x - minPoint.x << " height=" << maxPoint.z - minPoint.z << std::endl;
            else
                std::cout << "It is a cylinder!  diameter="<< (maxPoint.x - minPoint.x+maxPoint.y - minPoint.y)/4 << " height=" << maxPoint.z - minPoint.z << std::endl;
            std::cout << "===================================================================" << std::endl;

           
            

        }
            // viewer->addPointCloud<pcl::PointXYZRGB>(cloud_recentered2, "cloud2");
         std::cout<<"\n\n\n Total number of objects are "<< j<< "." <<std::endl;


  



  
  //--------------------
  // -----Main loop-----
  //--------------------
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
}