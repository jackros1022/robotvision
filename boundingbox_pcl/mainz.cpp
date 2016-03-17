#include <iostream>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/pca.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <boost/filesystem.hpp>
#include <pcl/common/io.h>
#include <pcl/filters/statistical_outlier_removal.h>

/*


#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <flann/flann.h>
#include <boost/filesystem.hpp>
#include <pcl/common/io.h>
*/

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
	return (viewer);
}


int
main(int argc, char** argv)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGB>);

	if (pcl::io::loadPCDFile<pcl::PointXYZRGB>("../learn15.pcd", *cloud) == -1) //* load the file
	{
		PCL_ERROR("Couldn't read file test_pcd.pcd \n");
		return (-1);
	}

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.01);
	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);

	if (inliers->indices.size() == 0)
	{
		PCL_ERROR("Could not estimate a planar model for the given dataset.");
		return (-1);
	}

	std::cerr << "Model coefficients: " << coefficients->values[0] << " "
		<< coefficients->values[1] << " "
		<< coefficients->values[2] << " "
		<< coefficients->values[3] << std::endl;

	pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	extract.setInputCloud(cloud);
	extract.setIndices(inliers);
	extract.setNegative(true);
	//Removes part_of_cloud but retain the original full_cloud
	//extract.setNegative(true); // Removes part_of_cloud from full cloud  and keep the rest
	extract.filter(*cloud_plane);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
	sor.setInputCloud(cloud_plane);
	sor.setMeanK(50);
	sor.setStddevMulThresh(1.0);
	sor.filter(*cloud_filtered);

	//cloud.swap(cloud_plane);
	/*
	pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
	viewer.showCloud(cloud_plane);
	while (!viewer.wasStopped())
	{
	}
	*/
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	Eigen::Matrix<float, 1, 3> fitted_plane_norm, xyaxis_plane_norm, rotation_axis;
	fitted_plane_norm[0] = coefficients->values[0];
	fitted_plane_norm[1] = coefficients->values[1];
	fitted_plane_norm[2] = coefficients->values[2];
	xyaxis_plane_norm[0] = 0.0;
	xyaxis_plane_norm[1] = 0.0;
	xyaxis_plane_norm[2] = 1.0;
	rotation_axis = xyaxis_plane_norm.cross(fitted_plane_norm);
	float theta = acos(xyaxis_plane_norm.dot(fitted_plane_norm)); 
	//float theta = -atan2(rotation_axis.norm(), xyaxis_plane_norm.dot(fitted_plane_norm));
	transform_2.rotate(Eigen::AngleAxisf(theta, rotation_axis));
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_recentered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::transformPointCloud(*cloud_filtered, *cloud_recentered, transform_2);

	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>); 
	pcl::copyPointCloud(*cloud_recentered, *cloud_xyz);

	///////////////////////////////////////////////////////////////////
	Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*cloud_xyz, pcaCentroid);
 	Eigen::Matrix3f covariance;
	
	computeCovarianceMatrixNormalized(*cloud_xyz, pcaCentroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	std::cout << eigenVectorsPCA.size() << std::endl;
	if(eigenVectorsPCA.size()!=9)
	eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));


	
	Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
	projectionTransform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
	projectionTransform.block<3, 1>(0, 3) = -1.f * (projectionTransform.block<3, 3>(0, 0) * pcaCentroid.head<3>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_xyz, *cloudPointsProjected, projectionTransform);
	// Get the minimum and maximum points of the transformed cloud.
	pcl::PointXYZ minPoint, maxPoint;
	pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
	const Eigen::Vector3f meanDiagonal = 0.5f*(maxPoint.getVector3fMap() + minPoint.getVector3fMap());



	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer = rgbVis(cloud);
	// viewer->addPointCloud<pcl::PointXYZ>(cloudPointsProjected, "Simple Cloud");
	// viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "Simple Cloud2");
	viewer->addCube(minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, minPoint.z , maxPoint.z);

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	
	/*
	pcl::PCA< pcl::PointXYZ > pca;
	pcl::PointCloud< pcl::PointXYZ >::ConstPtr cloud;
	pcl::PointCloud< pcl::PointXYZ > proj;

	pca.setInputCloud(cloud);
	pca.project(*cloud, proj);

	Eigen::Vector4f proj_min;
	Eigen::Vector4f proj_max;
	pcl::getMinMax3D(proj, proj_min, proj_max);

	pcl::PointXYZ min;
	pcl::PointXYZ max;
	pca.reconstruct(proj_min, min);
	pca.reconstruct(proj_max, max);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer->addCube(min.x, max.x, min.y, max.y, min.z, max.z);

	*/


	return (0);

}
