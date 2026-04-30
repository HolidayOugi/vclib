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
#include <limits>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

int moldCheck(
	vcl::PolyMesh          m,
	const std::vector<double>& gridCellSideLengths,
    bool debug
    ){
    // Configuration: cone angle in degrees for filtering
    const double CONE_ANGLE_DEGREES = 1.0;
	const double CONE_COS_THRESHOLD = std::cos(CONE_ANGLE_DEGREES * M_PI / 180.0);
    
    using namespace vcl;
    
    if (debug) {
        std::cout << "=== moldCheck started ===\n";
        std::cout.flush();
    }

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

    Point3d direction(1.0, 1.0, 0.0);

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
			return 1;
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
			return 1;
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

        
        // Mesh to collect hit points
        vcl::PolyMesh hitPointsMesh;
        std::vector<Point3d> allHitPoints(grid.rows * grid.cols);
		std::vector<uint> allHitFaceIds(grid.rows * grid.cols, UINT_NULL);
		std::vector<uint> allHitTriIds(grid.rows * grid.cols, UINT_NULL);
		std::vector<Point3d> allRayOrigins(grid.rows * grid.cols);
		std::vector<Point3d> allRayDirections(grid.rows * grid.cols);

        auto processCell = [&](uint idx) {
			uint j = idx / grid.cols; 
			uint i = idx % grid.cols; 

			Point3d               cellCenter;
			std::array<Point3d, 4> cellCorners;
			computeCellGeometry(i, j, cellCenter, cellCorners);

            // Shoot a single ray from -eps before the plane
            const Point3d rayOrigin = cellCenter + direction * (-EPS);
            
            allRayOrigins[idx] = rayOrigin;
            allRayDirections[idx] = direction;
            
            auto [faceId, baryCoords, triId] = 
                scene.firstFaceIntersectedByRay(rayOrigin, direction);
            
            if (faceId != UINT_NULL) {
                // Compute actual hit point using face vertices and barycentric coordinates
                const auto& face = m.face(faceId);
                
                // Get the triangulation for this face
                std::vector<uint> faceTriangulation = earCut(face);
                
                if (triId * 3 + 2 < faceTriangulation.size()) {
                    const uint vi0 = faceTriangulation[triId * 3 + 0];
                    const uint vi1 = faceTriangulation[triId * 3 + 1];
                    const uint vi2 = faceTriangulation[triId * 3 + 2];
                    
                    const Point3d& p0 = face.vertex(vi0)->position();
                    const Point3d& p1 = face.vertex(vi1)->position();
                    const Point3d& p2 = face.vertex(vi2)->position();
                    
                    Point3d hitPoint = 
                        p0 * baryCoords.x() + 
                        p1 * baryCoords.y() + 
                        p2 * baryCoords.z();
                    
                    allHitPoints[idx] = hitPoint;
					allHitFaceIds[idx] = faceId;
					allHitTriIds[idx] = triId;
                }
            }
        };
        

        std::vector<uint> allCells(grid.rows * grid.cols);
		std::iota(allCells.begin(), allCells.end(), 0);

        vcl::parallelFor(allCells, processCell);
        
        if (debug) {
            std::cout << "Ray casting complete. Hit points: ";
            uint hitCount = 0;
            for (uint i = 0; i < allHitPoints.size(); ++i) {
                if (allHitFaceIds[i] != UINT_NULL) ++hitCount;
            }
            std::cout << hitCount << "/" << allCells.size() << "\n";
            std::cout.flush();
        }

        // Lambda to check if a point is within a cone around another point's ray.
        auto isInCone = [](
            const Point3d& rayOrigin,
            const Point3d& rayDirection,
            const Point3d& testPoint,
			double coneCosThreshold) -> bool
        {
            const Point3d toPoint = testPoint - rayOrigin;
            const double projDist = toPoint.dot(rayDirection);
            if (projDist < 0) return false;  // Point is behind ray origin
            
            const double toPointNorm = toPoint.norm();
            if (toPointNorm < 1e-10) return true;  // Point coincides with origin
            
            const Point3d normalizedToPoint = toPoint / toPointNorm;
			const double cosAngle = normalizedToPoint.dot(rayDirection);
			return cosAngle >= coneCosThreshold;
        };

        // Filter points to keep only those without other points in their cone.
        // Use a vector to collect results in parallel (thread-safe per-element write)
        std::vector<bool> isFilteredPoints(allHitPoints.size(), false);
        std::vector<uint> filterIndices;
        filterIndices.reserve(allHitPoints.size());
        for (uint i = 0; i < allHitPoints.size(); ++i) {
            if (allHitFaceIds[i] != UINT_NULL) {
                filterIndices.push_back(i);
            }
        }
        
        if (debug) {
            std::cout << "Filtering " << filterIndices.size() << " points (parallel)...\n";
            std::cout.flush();
        }
        
        // Parallel filtering: each thread checks its point independently
        auto filterPoint = [&](uint idx) {
            uint i = filterIndices[idx];
            
            bool isFiltered = true;
            for (uint j = 0; j < allHitPoints.size(); ++j) {
                if (i == j || allHitFaceIds[j] == UINT_NULL) continue;
                
                // Check if point j is within the cone of point i
				if (isInCone(allRayOrigins[i], allRayDirections[i], allHitPoints[j], CONE_COS_THRESHOLD)) {
                    isFiltered = false;
                    break;
                }
            }
            
            isFilteredPoints[i] = isFiltered;
        };
        
        std::vector<uint> rangeIndices(filterIndices.size());
        std::iota(rangeIndices.begin(), rangeIndices.end(), 0);
        vcl::parallelFor(rangeIndices, filterPoint);
        
        // Build filtered set from results (single-threaded, no contention)
        std::set<std::pair<uint, uint>> filteredFaceTriPairs;
        for (uint i = 0; i < allHitPoints.size(); ++i) {
            if (isFilteredPoints[i] && allHitFaceIds[i] != UINT_NULL) {
                filteredFaceTriPairs.insert({allHitFaceIds[i], allHitTriIds[i]});
            }
        }
        
        if (debug) {
            std::cout << "Filtering complete. Found " << filteredFaceTriPairs.size() << " filtered points.\n";
            std::cout.flush();
        }

		if (debug) {
			// Create mesh from hit points (only non-zero hits).
			for (const auto& pt : allHitPoints) {
				if (pt != Point3d(0, 0, 0)) {
					hitPointsMesh.addVertex(pt);
				}
			}

			// Store hit triangles as explicit (faceId, triId) pairs for readability.
			std::set<std::pair<uint, uint>> hitTriangles;
			for (uint idx = 0; idx < allHitFaceIds.size(); ++idx) {
				if (allHitFaceIds[idx] != UINT_NULL && allHitTriIds[idx] != UINT_NULL) {
					hitTriangles.insert({allHitFaceIds[idx], allHitTriIds[idx]});
				}
			}

			// Build a triangulated copy of the input mesh for per-triangle coloring.
			vcl::TriMesh debugTriMesh;
			for (const auto& vtx : m.vertices()) {
				debugTriMesh.addVertex(vtx.position());
			}

			debugTriMesh.enablePerFaceColor();
			for (const auto& face : m.faces()) {
				const std::vector<uint> faceTriangulation = earCut(face);
				for (uint t = 0; t * 3 + 2 < faceTriangulation.size(); ++t) {
					const uint vi0 = face.vertex(faceTriangulation[t * 3 + 0])->index();
					const uint vi1 = face.vertex(faceTriangulation[t * 3 + 1])->index();
					const uint vi2 = face.vertex(faceTriangulation[t * 3 + 2])->index();

					const uint newFaceId = debugTriMesh.addFace(vi0, vi1, vi2);

					if (hitTriangles.contains({face.index(), t})) {
						debugTriMesh.face(newFaceId).color() = vcl::Color::Yellow;
					}
				}
			}

			// Build filtered mesh with red-colored triangles.
			vcl::TriMesh filteredTriMesh;
			for (const auto& vtx : m.vertices()) {
				filteredTriMesh.addVertex(vtx.position());
			}

			filteredTriMesh.enablePerFaceColor();
			for (const auto& face : m.faces()) {
				const std::vector<uint> faceTriangulation = earCut(face);
				for (uint t = 0; t * 3 + 2 < faceTriangulation.size(); ++t) {
					const uint vi0 = face.vertex(faceTriangulation[t * 3 + 0])->index();
					const uint vi1 = face.vertex(faceTriangulation[t * 3 + 1])->index();
					const uint vi2 = face.vertex(faceTriangulation[t * 3 + 2])->index();

					const uint newFaceId = filteredTriMesh.addFace(vi0, vi1, vi2);

					if (filteredFaceTriPairs.contains({face.index(), t})) {
						filteredTriMesh.face(newFaceId).color() = vcl::Color::Red;
					}
				}
			}

			// Build plane mesh as a quad spanning the projection bounds.
			vcl::TriMesh planeMesh;
			const double margin = 0.1 * std::max(lenU, lenV);
			const Point3d p0 = planePoint + u * (minU - margin) + v * (minV - margin);
			const Point3d p1 = planePoint + u * (maxU + margin) + v * (minV - margin);
			const Point3d p2 = planePoint + u * (maxU + margin) + v * (maxV + margin);
			const Point3d p3 = planePoint + u * (minU - margin) + v * (maxV + margin);

			const uint v0 = planeMesh.addVertex(p0);
			const uint v1 = planeMesh.addVertex(p1);
			const uint v2 = planeMesh.addVertex(p2);
			const uint v3 = planeMesh.addVertex(p3);

			planeMesh.addFace(v0, v1, v2);
			planeMesh.addFace(v0, v2, v3);

			const std::string base = std::string(VCLIB_RESULTS_PATH) + "/888_mold_check";
			saveMesh(hitPointsMesh, base + "_hit_points.ply");
			saveMesh(debugTriMesh, base + "_hit_triangles_yellow.ply");
			saveMesh(filteredTriMesh, base + "_filtered_red.ply");
			saveMesh(planeMesh, base + "_plane.ply");

			std::cout << "Found " << hitPointsMesh.vertexCount() << " hit points\n";
			std::cout << "Filtered points: " << filteredFaceTriPairs.size() << "\n";
			std::cout << "Saved debug meshes:\n"
					  << " - " << base << "_hit_points.ply\n"
					  << " - " << base << "_hit_triangles_yellow.ply\n"
					  << " - " << base << "_filtered_red.ply\n"
					  << " - " << base << "_plane.ply\n";
		}

        if (debug) {
            std::cout << "=== moldCheck completed successfully ===\n";
            std::cout.flush();
        }
        
        return 0;
}

int main()
{
    using namespace vcl;

    PolyMesh m = loadMesh<PolyMesh>(VCLIB_EXAMPLE_MESHES_PATH "/bunny_enlarged.ply");

    constexpr bool debug = true;

    std::vector<double> gridCellSideLengths = {0.3, 0.3};

    return moldCheck(std::move(m), gridCellSideLengths, debug);
}
