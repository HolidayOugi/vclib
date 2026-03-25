/*****************************************************************************
 * VCLib                                                                     *
 * Visual Computing Library                                                  *
 *                                                                           *
 * Copyright(C) 2021-2026                                                    *
 * Visual Computing Lab                                                      *
 * ISTI - Italian National Research Council                                  *
 *                                                                           *
 * All rights reserved.                                                      *
 *                                                                           *
 * This program is free software; you can redistribute it and/or modify      *
 * it under the terms of the Mozilla Public License Version 2.0 as published *
 * by the Mozilla Foundation; either version 2 of the License, or            *
 * (at your option) any later version.                                       *
 *                                                                           *
 * This program is distributed in the hope that it will be useful,           *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of            *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the              *
 * Mozilla Public License Version 2.0                                        *
 * (https://www.mozilla.org/en-US/MPL/2.0/) for more details.                *
 ****************************************************************************/

#include <vclib/embree/scene.h>
#include <vclib/algorithms/core/fibonacci.h>
#include <vclib/io.h>
#include <vclib/meshes.h>
#include <vclib/qt/mesh_viewer.h>
#include <vclib/render/drawable/drawable_mesh.h>

#include <QApplication>
#include <cmath>
#include <limits>
#include <iostream>
#include <tuple>

class MeshViewerSelectQt : public vcl::qt::MeshViewer
{
public:
    using vcl::qt::MeshViewer::MeshViewer;

    MeshViewerSelectQt(QWidget* parent = nullptr) : vcl::qt::MeshViewer(parent)
    {
        viewer().setOnObjectSelected([this](uint id) {
            drawableObjectVectorTree().setSelectedItem(id);
        });
    }
};

int main(int argc, char** argv)
{
    //
}
