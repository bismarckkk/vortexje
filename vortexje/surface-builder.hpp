//
// Vortexje -- Surface builder.
//
// Copyright (C) 2012 - 2014 Baayen & Heinz GmbH.
//
// Authors: Jorn Baayen <jorn.baayen@baayen-heinz.com>
//

#ifndef __SURFACE_BUILDER_HPP__
#define __SURFACE_BUILDER_HPP__

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <vortexje/lifting-surface.hpp>

namespace Vortexje
{

/**
   Class for construction of surfaces by joining 2-D shapes together.
   
   @brief  surface construction helper.
*/
class SurfaceBuilder
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    SurfaceBuilder(Surface &surface);
    
    /**
       Surface under construction.
    */
    Surface &surface;
    
    std::vector<int> create_nodes(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &points);
    
    std::vector<int> create_panels_between(const std::vector<int> &first_nodes, const std::vector<int> &second_nodes, bool cyclic = true);
    
    std::vector<int> create_panels_inside(const std::vector<int> &nodes, const Eigen::Vector3d &tip_point, int z_sign);
    
    void finish();
};

};

#endif // __SURFACE_BUILDER_HPP__
