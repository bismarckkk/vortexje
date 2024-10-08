//
// Vortexje -- Wake.
//
// Copyright (C) 2012 - 2014 Baayen & Heinz GmbH.
//
// Authors: Jorn Baayen <jorn.baayen@baayen-heinz.com>
//

#ifndef __WAKE_HPP__
#define __WAKE_HPP__

#include <memory>

#include <Eigen/Geometry>

#include <vortexje/lifting-surface.hpp>

namespace Vortexje
{

/**
   Representation of a wake, i.e., a vortex sheet.
   
   @brief Wake representation.
*/
class Wake : public Surface
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Wake(std::shared_ptr<LiftingSurface> lifting_surface);
    
    /**
       Associated lifting surface.
    */
    std::shared_ptr<LiftingSurface> lifting_surface;
    
    virtual void add_layer();
    void remove_layer();
    
    void translate_trailing_edge(const Eigen::Vector3d &translation);
    void transform_trailing_edge(const Eigen::Transform<double, 3, Eigen::Affine> &transformation);
    
    virtual void update_properties(double dt);
    
    virtual Eigen::Vector3d vortex_ring_unit_velocity(const Eigen::Vector3d &x, int this_panel) const;
    
    /**
       Strengths of the doublet, or vortex ring, panels.
    */
    std::vector<double> doublet_coefficients;

    std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> get_vortex_particles();
    std::vector<double> last_doublet_coefficients;
};

};

#endif // __WAKE_HPP__
