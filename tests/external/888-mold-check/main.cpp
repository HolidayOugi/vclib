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
#include <vclib/io.h>
#include <vclib/meshes.h>

#include <cmath>

int moldCheck(
	vcl::PolyMesh          m,
	const std::vector<double>& gridCellSideLengths,
    bool debug
    ){
    
    using namespace vcl;

    struct GridChoice
	{
		uint rows = 1;
		uint cols = 1;
		double sideU = 0.0;
		double sideV = 0.0;
	};

    auto chooseGrid = [&](double lenU, double lenV) -> GridChoice {
		if (lenU <= 0.0 || lenV <= 0.0) {
			return {1, 1, lenU, lenV};
		}

		const double sideU =
			(gridCellSideLengths.size() >= 1) ? gridCellSideLengths[0] : lenU;
		const double sideV =
			(gridCellSideLengths.size() >= 2) ? gridCellSideLengths[1] : sideU;

		if (sideU <= 0.0 || sideV <= 0.0) {
			return {1, 1, lenU, lenV};
		}

		const uint cols = static_cast<uint>(std::max(1.0, std::ceil(lenU / sideU)));
		const uint rows = static_cast<uint>(std::max(1.0, std::ceil(lenV / sideV)));

		return {rows, cols, sideU, sideV};
	};


    updateBoundingBox(m);

    const double EPS = 1e-6 * m.boundingBox().diagonal();

    embree::Scene scene(m);

    Point3d direction(0.0, 0.5, 0.0);

    direction.normalize();

    double minProj = std::numeric_limits<double>::infinity();
		for (const auto& vv : m.vertices()) {
			minProj = std::min(minProj, vv.position().dot(direction));
		}
	
    const Point3d planePoint = direction * minProj;
	const Planed  plane(planePoint, direction);

    Point3d u, v;
		direction.orthoBase(u, v);
		if (u.norm() <= EPS || v.norm() <= EPS) {
			return std::numeric_limits<double>::infinity();
		}
		u.normalize();
		v.normalize();

		double minU = std::numeric_limits<double>::infinity();
		double minV = std::numeric_limits<double>::infinity();
		double maxU = -std::numeric_limits<double>::infinity();
		double maxV = -std::numeric_limits<double>::infinity();

		std::vector<Point3d> projectedPoints;

        if (debug) {
            projectedPoints.reserve(
				std::distance(m.vertices().begin(), m.vertices().end()));
        }

		for (const auto& vert : m.vertices()) {
			const Point3d projected = plane.projectPoint(vert.position());
			const Point3d rel = projected - planePoint;

			const double pu = rel.dot(u);
			const double pv = rel.dot(v);

			minU = std::min(minU, pu);
			minV = std::min(minV, pv);
			maxU = std::max(maxU, pu);
			maxV = std::max(maxV, pv);
		}

		const double lenU = maxU - minU;
		const double lenV = maxV - minV;
		if (lenU <= EPS || lenV <= EPS) {
			return std::numeric_limits<double>::infinity();
		}

		const GridChoice grid = chooseGrid(lenU, lenV);

        const double cellDu   = grid.sideU;
		const double cellDv   = grid.sideV;
		const double cellArea = cellDu * cellDv;

		auto computeCellGeometry = [&](uint i,
								 uint j,
								 Point3d& cellCenter,
								 std::array<Point3d, 4>& cellCorners) {
			const double u0 = minU + i * cellDu;
			const double u1 = u0 + cellDu;
			const double v0 = minV + j * cellDv;
			const double v1 = v0 + cellDv;

			cellCorners = {
				planePoint + u * u0 + v * v0,
				planePoint + u * u1 + v * v0,
				planePoint + u * u1 + v * v1,
				planePoint + u * u0 + v * v1};

			const double centerU = minU + (i + 0.5) * cellDu;
			const double centerV = minV + (j + 0.5) * cellDv;
			cellCenter           = planePoint + u * centerU + v * centerV;
		};

        
        auto processCell = [&](uint i, uint j) {
			Point3d               cellCenter;
			std::array<Point3d, 4> cellCorners;
			computeCellGeometry(i, j, cellCenter, cellCorners);


            //add points
        };
        

        std::vector<uint> allCells(grid.rows * grid.cols);
		std::iota(allCells.begin(), allCells.end(), 0);

        vcl::parallelFor(allCells, [&](uint idx) {
				uint j = idx / grid.cols; 
				uint i = idx % grid.cols; 
				processCell(i, j);
    });


    


    return 0;
}

int main()
{
    using namespace vcl;

    PolyMesh m = loadMesh<PolyMesh>(VCLIB_EXAMPLE_MESHES_PATH "/bunny.obj");

    constexpr bool debug = true;

    std::vector<double> gridCellSideLengths = {0.1, 0.1};

    return moldCheck(std::move(m), gridCellSideLengths, debug);
}
