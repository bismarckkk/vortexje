/************************************************
 * Copyright (c) 2025. Zuo Qingyu               *
 * All rights reserved.                         *
 ************************************************/

#ifndef HELIVPM_XDMF_SURFACE_WRITER_HPP
#define HELIVPM_XDMF_SURFACE_WRITER_HPP

#include <string>

#include <vortexje/surface-writer.hpp>

namespace Vortexje
{

    class XdmfSurfaceWriter : public SurfaceWriter
    {
    public:
        const char *file_extension() const;

        bool write(const std::shared_ptr<Surface> &surface, const std::string &_filename,
                   int node_offset, int panel_offset,
                   const std::vector<std::string> &view_names, const std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd> > &view_data);
    };

};

#endif //HELIVPM_XDMF_SURFACE_WRITER_HPP
