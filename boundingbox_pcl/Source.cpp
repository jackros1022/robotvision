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


#include <pcl/filters/voxel_grid.h>
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
#include <pcl/common/distances.h>
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
printUsage(const char* progName)
{
	std::cout << "\n\nUsage: " << progName << " [options]\n\n"
		<< "Options:\n"
		<< "-------------------------------------------\n"
		<< "-h           this help\n"
		<< "-1           Solution_1\n"
		<< "-2           Solution_2\n"
		<< "-3           Solution_3\n"
		<< "\n\n";
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	viewer->setCameraPosition(0.0, -1.8, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
	return (viewer);
}


unsigned int text_id = 0;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
	void* viewer_void)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
	if (event.getKeySym() == "r" && event.keyDown())
	{
		std::cout << "r was pressed => removing all text" << std::endl;

		char str[512];
		for (unsigned int i = 0; i < text_id; ++i)
		{
			sprintf(str, "text#%03d", i);
			viewer->removeShape(str);
		}
		text_id = 0;
	}
}

void mouseEventOccurred(const pcl::visualization::MouseEvent &event,
	void* viewer_void)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
	if (event.getButton() == pcl::visualization::MouseEvent::LeftButton &&
		event.getType() == pcl::visualization::MouseEvent::MouseButtonRelease)
	{
		std::cout << "Left mouse button released at position (" << event.getX() << ", " << event.getY() << ")" << std::endl;

		char str[512];
		sprintf(str, "text#%03d", text_id++);
		viewer->addText("clicked here", event.getX(), event.getY(), str);
	}
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> interactionCustomizationVis()
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);

	viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);
	viewer->registerMouseCallback(mouseEventOccurred, (void*)&viewer);

	return (viewer);
}





// --------------
// -----Main-----
// --------------
int
main(int argc, char** argv)
{
	// --------------------------------------
	// -----Parse Command Line Arguments-----
	// --------------------------------------
	if (pcl::console::find_argument(argc, argv, "-h") >= 0)
	{
		printUsage(argv[0]);
		return 0;
	}

	int Solution_numb = 0;
	if (pcl::console::find_argument(argc, argv, "-1") >= 0)
	{
		Solution_numb = 1;
	}
	else if (pcl::console::find_argument(argc, argv, "-2") >= 0)
	{
		Solution_numb = 2;
	}
	else if (pcl::console::find_argument(argc, argv, "-3") >= 0)
	{
		Solution_numb = 3;
	}
	else
	{
		printUsage(argv[0]);
		//return 0;
	}


	pcl::PCDReader reader;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>), cloud_f(new pcl::PointCloud<pcl::PointXYZRGB>);
    // Create pcl visualizer
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	// Read in the cloud data
	reader.read(argv[1], *cloud_ptr);
	std::cout << "PointCloud before filtering has: " << cloud_ptr->points.size() << " data points." << std::endl; //*

	// Create the filtering object: down_sample the dataset using a leaf size of 0.005f
	pcl::VoxelGrid<pcl::PointXYZRGB> vg;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>), cloud_objects(new pcl::PointCloud<pcl::PointXYZRGB>);;
	vg.setInputCloud(cloud_ptr);
	vg.setLeafSize(0.005f, 0.005f, 0.005f);
	vg.filter(*cloud_filtered);
	std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl; //*


	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PCDWriter writer;

	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(150);
	seg.setDistanceThreshold(0.01);

	int i = 0, nr_points = (int)cloud_filtered->points.size();

	// Segment the largest planar component 
	seg.setInputCloud(cloud_filtered);
	seg.segment(*inliers, *coefficients);
	// Print out the plane coefficients
	std::cout << "Model coefficients: "
		<< coefficients->values[0] << " "
		<< coefficients->values[1] << " "
		<< coefficients->values[2] << " "
		<< coefficients->values[3] << std::endl;

	// Extract the planar inliers from the input cloud
	pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	extract.setInputCloud(cloud_filtered);
	extract.setIndices(inliers);
	extract.setNegative(false);

	// Get the points associated with the planar surface
	extract.filter(*cloud_plane);
	std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << std::endl;

	// Get the centroid of the fitted plane which is used for translation later.  
	Eigen::Vector4f plane_centroid;
	pcl::compute3DCentroid(*cloud_plane, plane_centroid);
	Eigen::Vector3f plane_translation;
	plane_translation.head(3) << plane_centroid[0], plane_centroid[1], plane_centroid[2];

	// Remove the planar inliers, extract the rest of point cloud
	extract.setNegative(true);
	extract.filter(*cloud_f);
	*cloud_objects = *cloud_f;


	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_objects_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
	sor.setInputCloud(cloud_objects);
	sor.setMeanK(50);
	sor.setStddevMulThresh(1.0);
	sor.filter(*cloud_objects_filtered);


	// Rotate and translate the point cloud to align the plane normal of point cloud to world z axis.  
	Eigen::Affine3f transform_1 = Eigen::Affine3f::Identity();
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	Eigen::Matrix<float, 1, 3> fitted_plane_norm, xyaxis_plane_norm, rotation_axis;
	fitted_plane_norm[0] = coefficients->values[0];
	fitted_plane_norm[1] = coefficients->values[1];
	fitted_plane_norm[2] = coefficients->values[2];
	xyaxis_plane_norm[0] = 0.0;
	xyaxis_plane_norm[1] = 0.0;
	xyaxis_plane_norm[2] = 1.0;
	rotation_axis = xyaxis_plane_norm.cross(fitted_plane_norm);
	rotation_axis = rotation_axis.normalized();
	float theta = -acos(xyaxis_plane_norm.dot(fitted_plane_norm));
	// rotation matrix
	transform_2.rotate(Eigen::AngleAxisf(theta, rotation_axis));
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr objects_recentered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sence_recentered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr objects_recentered_t(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sence_recentered_t(new pcl::PointCloud<pcl::PointXYZRGB>);
	// translation matrix
	transform_1.translation() = -plane_translation;
	//translate first and then rotation 
	pcl::transformPointCloud(*cloud_objects_filtered, *objects_recentered_t, transform_1);
	pcl::transformPointCloud(*cloud_ptr, *sence_recentered_t, transform_1);

	pcl::transformPointCloud(*objects_recentered_t, *objects_recentered, transform_2);
	pcl::transformPointCloud(*sence_recentered_t, *sence_recentered, transform_2);

	// show transformed point cloud
	viewer = rgbVis(sence_recentered);

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud(objects_recentered);

	std::vector<pcl::PointIndices> cluster_indices;
	// Clustering 
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	ec.setClusterTolerance(0.02); // 0.2cm
	ec.setMinClusterSize(100);
	ec.setMaxClusterSize(25000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(objects_recentered);
	ec.extract(cluster_indices);

	// Vectors used for storing clustered point clouds which are used for later processing
	vector<Eigen::Vector4f> cluster_centers;
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_pclouds;
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> final_cluster_pclouds;
	vector<float> cluster_heights;



	// Store information(max height, point cloud, centroid) of each cluster result from EuclideanClusterExtraction(ec), use this information to 
	//combine two adjacent clusters which might be belong into one cluster. 
	
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
			cloud_cluster->points.push_back(objects_recentered->points[*pit]); //*
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;


		//std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
		
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_xyz(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(*cloud_cluster, *cloud_cluster_xyz);
		// Outliers filter, smooth cluster.
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor_obj;
		sor_obj.setInputCloud(cloud_cluster_xyz);
		sor_obj.setMeanK(30);
		sor_obj.setStddevMulThresh(1.5);
		sor_obj.filter(*cloud_cluster_filtered);

		//std::cout << "After Outliers Removal, Object Point Cloud: " << cloud_cluster_filtered->points.size() << " data points." << std::endl;
		// Storing the information of each cluster from EuclideanClusterExtraction(ec)
		Eigen::Vector4f clusterCentroid;
		pcl::compute3DCentroid(*cloud_cluster_filtered, clusterCentroid);
		cluster_centers.push_back(clusterCentroid);
		cluster_pclouds.push_back(cloud_cluster_filtered);

		pcl::PointXYZ mmaxforh, mminforh;
		pcl::getMinMax3D(*cloud_cluster_filtered, mminforh, mmaxforh);
		cluster_heights.push_back(mmaxforh.z);

	}
	

	// Combine two adjacent clusters which satisfy our two constraints: 1. The euler distance between their centroid is less than 0.12; 2. heigth difference is less than 0.005 
	// 
	Redo:
	for (i = 0; i < cluster_centers.size(); i++)
	{
		//bool combineflag = false;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		for (int j = i + 1; j < cluster_centers.size(); j++)
		{
			Eigen::Vector4f centroiddis = cluster_centers[i] - cluster_centers[j];
			float centroid_distance = sqrt(centroiddis[0] * centroiddis[0] + centroiddis[1] * centroiddis[1] + centroiddis[2] * centroiddis[2]);
			//cout << centroid_distance << endl;
			//cout << "height : " << cluster_heights[i] << "    " << cluster_heights[j] << endl;

			if (centroid_distance < 0.12 && abs(cluster_heights[i] - cluster_heights[j])< 0.005)
			{	
				*cloud_cluster_filtered = *cluster_pclouds[i] + *cluster_pclouds[j];
				Eigen::Vector4f new_center = 0.5f*(cluster_centers[i] + cluster_centers[j]);
				float new_height = 0.5f*(cluster_heights[i] + cluster_heights[j]);

				cluster_pclouds.erase(cluster_pclouds.begin() + i);
				cluster_pclouds.erase(cluster_pclouds.begin() + j-1);

				cluster_centers.erase(cluster_centers.begin() + i);
				cluster_centers.erase(cluster_centers.begin() + j-1);
				
				cluster_heights.erase(cluster_heights.begin() + i);
				cluster_heights.erase(cluster_heights.begin() + j-1);

				cluster_pclouds.push_back(cloud_cluster_filtered);
				cluster_centers.push_back(new_center);
				cluster_heights.push_back(new_height);
				goto Redo;
			}
			
		}

	}


	// Boolean variable: cylinder: true ; box: faulse, used for question 3.
	bool cylindersOrBoxes = true;
	int cylinder_counter = 0, box_counter =0 ;
	int mcounter = 0;

	// Solution 2:
	// We create oriented bounding box by PCA for each cluster, 
	// Because we have assumption that boxes/cylinders are standing "upright", we can fix one eigenvector calculated from covariance matrix in PCA with [0,0,1](z axis)
	// and use second biggest eigenvector as the second direction, use cross product to find the third direction
	
	vector<float> object_width;
	vector<float> object_length;
	vector<float> object_height;
	
	for (int mi = 0; mi < cluster_pclouds.size(); mi++)
	{
        
		pcl::PointCloud<pcl::PointXYZ>::Ptr one_object_xyz(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(*(cluster_pclouds[mcounter]), *one_object_xyz);

		// PCA processing as follow:
		Eigen::Vector4f objCentroid;
		pcl::compute3DCentroid(*one_object_xyz, objCentroid);
		
		Eigen::Matrix3f covariance;
		// Get our covariance matrix
		computeCovarianceMatrixNormalized(*one_object_xyz, objCentroid, covariance);
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
		// Calculate eigenvectors of the covariance matrix, which are the directions for the cluster (in other words, three eigenvectors set up a new coordinate system)
		Eigen::Matrix3f meigenVectors = eigen_solver.eigenvectors();
		
        // Manually assign z axis as one of PCA direction, since we have a assumption that boxes/cylinders are standing "upright".
		Eigen::Matrix<float, 3, 1> z_axis;
		z_axis[0] = 0.0;
		z_axis[1] = 0.0;
		z_axis[2] = 1.0;
		meigenVectors.col(2) = z_axis;
       
		// Find out other two directions
		meigenVectors.col(1) = meigenVectors.col(0).cross(meigenVectors.col(2));
		meigenVectors.col(1) = meigenVectors.col(1).normalized();
		meigenVectors.col(0) = meigenVectors.col(1).cross(meigenVectors.col(2));
		meigenVectors.col(0) = meigenVectors.col(0).normalized();

		// Find transformation from new coordinate system got from PCA to origin coordinate system 
		Eigen::Matrix4f TransformationNtoO(Eigen::Matrix4f::Identity());
		TransformationNtoO.block<3, 3>(0, 0) = meigenVectors.transpose();
		TransformationNtoO.block<3, 1>(0, 3) = -1.f * (TransformationNtoO.block<3, 3>(0, 0) * objCentroid.head<3>());

		// Transform object to origin coordinate system to find out max/min bounding box through getMinMax3D function. 
		pcl::PointCloud<pcl::PointXYZ>::Ptr ReprojectObj(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(*one_object_xyz, *ReprojectObj, TransformationNtoO);

		// Get the minimum and maximum bounding box.
		pcl::PointXYZ minPoint, maxPoint;
		pcl::getMinMax3D(*ReprojectObj, minPoint, maxPoint);

		
		const Eigen::Vector3f meanDiagonal = 0.5f*(maxPoint.getVector3fMap() + minPoint.getVector3fMap());
		//find transformation from origin coordinate system to the new coordinate system to transform calculated bounding box onto object.
		const Eigen::Quaternionf Quaternion_boundingbox(meigenVectors);
		const Eigen::Vector3f Translation_boundingbox = meigenVectors * meanDiagonal + objCentroid.head<3>();

		// Add bounding box 
		viewer->addCube(Translation_boundingbox, Quaternion_boundingbox, maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z, "minmax_" + std::to_string(mcounter));
	
		object_width.push_back(maxPoint.x - minPoint.x);
		object_length.push_back(maxPoint.y - minPoint.y);
		object_height.push_back(maxPoint.z - minPoint.z);
		//Solution 3:
        //Because we use PCA to create oriented bounding box, one step in PCA can let us rotate the point cloud to origin coordinate (the step to find max/min 3D points),
        //at this step, we can check whether the cluster is cylinder or not.
        // Since we align the plane normal to z axis, we can ignore information from z axis, things become to 2D, 
		// We use two constraint to judge whether the cluster is cylinder or not,   
		// The first one is that if the ratio between max(boundingbox_width, boundingbox_length) and min(boundingbox_width, boundingbox_length) is greater than 1.27f
		// it means the cluster is cuboid.
		// The second constraint is, if the ratio is less than 1.27f, we project all 3D points into xy_plane, and draw a cycle, the radius of cycle is max(boundingbox_width, boundingbox_length)/2
		// the center of cycle is [ 0.5f*(minPoint.x + maxPoint.x), minPoint.y + maxdis ], check whether most points in cluster are inside the cycle
		// If most points are inside the cycle, it means the cluster is cylinder, otherwise it means it is box.
		
			float bboxwidth = maxPoint.x - minPoint.x;
			float bboxlength = maxPoint.y - minPoint.y;
			float mmax = max(bboxwidth, bboxlength);
			float mmin = min(bboxwidth, bboxlength);
			float maxdis = mmax*0.5f;
			int inliers_counter = 0;
			//Eigen::Vector2f cycle_center(0.5f*(minPoint.x + maxPoint.x), 0.5f*(minPoint.y + maxPoint.y));
			Eigen::Vector2f cycle_center(0.5f*(minPoint.x + maxPoint.x), minPoint.y + maxdis);
			float mratio = mmax / mmin;
			
			//cout << mratio << endl;
			if (mratio > 1.27f)
			{
				box_counter++;
			}
			else{

				for (size_t j = 0; j < ReprojectObj->points.size(); j++)
				{
					if ((ReprojectObj->points[j].x - cycle_center[0]) *(ReprojectObj->points[j].x - cycle_center[0])
						+ (ReprojectObj->points[j].y - cycle_center[1]) *(ReprojectObj->points[j].y - cycle_center[1]) <= maxdis*maxdis)
						inliers_counter++;

				}
				float inliers_ratio = inliers_counter / (float)ReprojectObj->points.size();
			//	cout << "inliers_ratio " << inliers_ratio << endl;
				if (inliers_ratio > 0.899)
				{
					cylinder_counter++;
				}
				else
					box_counter++;
			}
			
				mcounter++;

	}
	cylindersOrBoxes = box_counter >= cylinder_counter ? false : true;



	switch (Solution_numb)
	{
	case 0:
		cout << "Please use -1,-2,-3 in the command arguments to check answer 1,2,3 " << endl;
		break;
	case 1:

		cout << "Answer 1: " << endl;
		cout << "The number of objects is " << cluster_pclouds.size() << endl;
		break;
	case 2:

		cout << "Answer 2: " << endl;
		if (!cylindersOrBoxes)
		{
			cout << "There are " << cluster_pclouds.size() <<  " boxes on the table " << endl;
			for (int j = 0; j < object_width.size(); j++)
				cout << " Parameters (height x width x length) for the \"" << j + 1 << "\" box are "
				<< object_height[j] << " x " << object_width[j] << " x " << object_length[j] << endl;
		}
		else
		{
			cout << "There are " << cluster_pclouds.size() << " cylinders on the table " << endl;
			for (int j = 0; j < object_width.size(); j++)
				cout << " Parameters (radius, height) for the \"" << j + 1 << "\" cylinder are "
				<< "( " << object_width[j] / 2 << ", " << object_height[j] << " ) " << endl;
		}
		break;
	case 3:
		cout << "Answer 3: " << endl;
		if (!cylindersOrBoxes)
		{
			cout << "There are " << cluster_pclouds.size() << " boxes on the table " << endl;
			
		}
		else
		{
			cout << "There are " << cluster_pclouds.size() << " cylinders on the table " << endl;
			
		}
		break;
	}

   // Main loop for visualizer
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}



}
