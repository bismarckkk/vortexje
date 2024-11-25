//
// Vortexje -- Lifting surface.
//
// Copyright (C) 2012 - 2014 Baayen & Heinz GmbH.
//
// Authors: Jorn Baayen <jorn.baayen@baayen-heinz.com>
//

#ifndef __LIFTING_SURFACE_HPP__
#define __LIFTING_SURFACE_HPP__

#include <Eigen/Core>

#include <vortexje/surface.hpp>

namespace Vortexje
{

/**
   Representation of lifting surface.
   
   Note:  The labels "upper" and "lower" must be consistent within the same surface, but are otherwise arbitrary.
   
   @brief Lifting surface representation.
*/
class LiftingSurface : public Surface
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    LiftingSurface(const std::string &id);
         
    /**
       Nodes on the upper side of the surface.  The first dimension is the chordwise direction;  the second the spanwise direction.
       
       Note:  This matrix must at least cover the nodes belonging to the upper side panels adjacent to the trailing edge.
    */
    Eigen::MatrixXi upper_nodes;
    
    /**
       Nodes on the lower side of the surface.  The first dimension is the chordwise direction;  the second the spanwise direction.
       
       Note:  This matrix must at least cover the nodes belonging to the lower side panels adjacent to the trailing edge.
    */
    Eigen::MatrixXi lower_nodes;
    
    /**
       Panels on the upper side of the surface.  The first dimension is the chordwise direction;  the second the spanwise direction.
       
       Note:  This matrix must at least cover the upper side panels adjacent to the trailing edge.
    */
    Eigen::MatrixXi upper_panels;
    
    /**
       Panels on the lower side of the surface.  The first dimension is the chordwise direction;  the second the spanwise direction.
       
       Note:  This matrix must at least cover the lower side panels adjacent to the trailing edge.
    */
    Eigen::MatrixXi lower_panels;

    Eigen::Vector3d lift_dir=Eigen::Vector3d::Zero(), drag_dir=Eigen::Vector3d::Zero();
    std::vector<double> dx;

    int vpNumberPerStep;
    
    int n_chordwise_nodes() const;
    int n_chordwise_panels() const;
    
    int n_spanwise_nodes() const;
    int n_spanwise_panels() const;
    
    int trailing_edge_node(int index) const;
    int trailing_edge_upper_panel(int index) const;
    int trailing_edge_lower_panel(int index) const;
    
    void finish_trailing_edge();
    
    virtual void transform(const Eigen::Transform<double, 3, Eigen::Affine> &transformation);
    
    virtual Eigen::Vector3d wake_emission_velocity(const Eigen::Vector3d &apparent_velocity, int node_index) const;
    
private:
    /**
       Cached list of trailing edge bisector vectors.
    */
    Eigen::MatrixXd trailing_edge_bisectors;
    
    /**
       Cached list of vectors normal to the initial wake strip surface.
    */
    Eigen::MatrixXd wake_normals;
};

};

#endif // __LIFTING_SURFACE_HPP__
