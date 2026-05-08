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
#include <vector>

int moldCheck(
	vcl::PolyMesh              m,
	const std::vector<double>& gridCellSideLengths,
	bool                       debug)
{
	using namespace vcl;

	// Configuration: cone angle in degrees for filtering
	const double CONE_ANGLE_DEGREES = 1.0;
	const double CONE_COS_THRESHOLD = std::cos(CONE_ANGLE_DEGREES * M_PI / 180.0);
    
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

	const double MAX_DISTANCE = m.boundingBox().diagonal();
	const double EPS = 1e-12 * MAX_DISTANCE;

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


		for (const auto& vert : m.vertices()) {
			const Point3d projected = plane.projectPoint(vert.position());
			const Point3d rel = projected - planePoint;

			const double pu = rel.dot(u);
			const double pv = rel.dot(v);

			minU = std::min(minU, pu - 0.5 * MAX_DISTANCE);
			minV = std::min(minV, pv - 0.5 * MAX_DISTANCE);
			maxU = std::max(maxU, pu + 0.5 * MAX_DISTANCE);
			maxV = std::max(maxV, pv + 0.5 * MAX_DISTANCE);
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


		struct CellData
		{
			double distance;
			Point3d hitPoint;
			bool hasHit = false;
		};

		std::vector<CellData> cells(grid.rows * grid.cols);

		auto computeCell = [&](uint idx) -> CellData {
			uint j = idx / grid.cols; 
			uint i = idx % grid.cols; 

			Point3d               cellCenter;
			std::array<Point3d, 4> cellCorners;
			computeCellGeometry(i, j, cellCenter, cellCorners);

			// Shoot a single ray from -eps before the plane
			const Point3d rayOrigin = cellCenter + direction * (-EPS);
			const Point3d invalidPoint = cellCenter + direction * MAX_DISTANCE;
            
            auto [faceId, baryCoords, triId] = 
                scene.firstFaceIntersectedByRay(rayOrigin, direction);
            
			if (faceId != UINT_NULL) {
				// Compute actual hit point using face vertices and barycentric coordinates
                const auto& face = m.face(faceId);
				std::vector<uint> faceTriangulation = earCut(face);

				if (triId * 3 + 2 < faceTriangulation.size()) {
					const uint vi0 = faceTriangulation[triId * 3 + 0];
					const uint vi1 = faceTriangulation[triId * 3 + 1];
					const uint vi2 = faceTriangulation[triId * 3 + 2];

					const Point3d& p0 = face.vertex(vi0)->position();
					const Point3d& p1 = face.vertex(vi1)->position();
					const Point3d& p2 = face.vertex(vi2)->position();

					const Point3d hitPoint =
						p0 * baryCoords.x() + p1 * baryCoords.y() + p2 * baryCoords.z();

					const double distance = std::abs((hitPoint - planePoint).dot(direction));
					return CellData{distance, hitPoint, true};
				}
            }

			return CellData{MAX_DISTANCE, invalidPoint, false};
        };
        

        std::vector<uint> allCells(grid.rows * grid.cols);
		std::iota(allCells.begin(), allCells.end(), 0);

		vcl::parallelFor(allCells, [&](uint idx) {
			cells[idx] = computeCell(idx);
		});

		std::vector<CellData> clampedCells = cells;

        
        if (debug) {
            std::cout << "Ray casting complete. Hit points: ";
            uint hitCount = 0;
			for (uint i = 0; i < cells.size(); ++i) {
				if (cells[i].distance < MAX_DISTANCE) {
                    ++hitCount;
                }
            }
            std::cout << hitCount << "/" << allCells.size() << " cells \n";
            std::cout.flush();
        }

		auto isWithinPlaneAngle = [&](
			const Point3d& point,
			const Point3d& other,
			double         coneCosThreshold) -> bool
		{
			const Point3d directionToOther = other - point;
			const double  directionToOtherNorm = directionToOther.norm();
			if (directionToOtherNorm < EPS) return true;

			// Il cono si apre verso il piano, quindi nella direzione -direction
			const Point3d dirToPlane = -direction;
			
			const Point3d dirToOtherNormalized = directionToOther / directionToOtherNorm;
			const double  cosBetween = dirToOtherNormalized.dot(dirToPlane);
			return cosBetween > coneCosThreshold - EPS; // Usa -EPS invece di +EPS per essere più permissivo
		};

		auto coneBoundaryStep = [&](const Point3d& a, const Point3d& b) -> double {
			// Vogliamo che il punto b sia FUORI dal cono di a
			// Il cono di a ha vertice in a e si apre nella direzione -direction
			// Vogliamo: angle(b - new_a, -direction) >= threshold
			// dove new_a = a - t*direction
			
			const Point3d ab = b - a;
			
			// Se b è già fuori dal cono (angolo sufficientemente grande), t=0
			const double ab_norm = ab.norm();
			if (ab_norm < EPS) return 0.0;
			
			const double cos_angle = (ab / ab_norm).dot(-direction);
			if (cos_angle <= CONE_COS_THRESHOLD) {
				return 0.0; // b è già fuori dal cono
			}
			
			// Dobbiamo trovare t tale che:
			// dot((b - (a - t*d)) / |b - (a - t*d)|, -d) = cos_threshold
			// Dove d = direction
			
			// Sia v = b - a, d = direction
			// new_v = v + t*d
			// Vogliamo: dot(new_v / |new_v|, -d) = cos_threshold
			
			const Point3d v = b - a;
			const double v_dot_d = v.dot(direction);
			const double v_norm2 = v.dot(v);
			
			// L'equazione è: -(v_dot_d + t) / sqrt(v_norm2 + 2*t*v_dot_d + t²) = cos_threshold
			// => -(v_dot_d + t) = cos_threshold * sqrt(v_norm2 + 2*t*v_dot_d + t²)
			// Quadrando: (v_dot_d + t)² = cos²_threshold * (v_norm2 + 2*t*v_dot_d + t²)
			
			const double cos2 = CONE_COS_THRESHOLD * CONE_COS_THRESHOLD;
			const double sin2 = 1.0 - cos2;
			
			// sin² * t² + 2*(v_dot_d - cos²*v_dot_d)*t + (v_dot_d² - cos²*v_norm2) = 0
			// sin² * t² + 2*v_dot_d*sin²*t + (v_dot_d² - cos²*v_norm2) = 0
			
			const double a_eq = sin2;
			const double b_eq = 2.0 * v_dot_d * sin2;
			const double c_eq = v_dot_d * v_dot_d - cos2 * v_norm2;
			
			const double discriminant = b_eq * b_eq - 4.0 * a_eq * c_eq;
			
			if (discriminant <= 0.0) {
				// Nessuna soluzione reale, sposta completamente
				return v_dot_d; // Sposta abbastanza da allineare con il piano
			}
			
			const double sqrt_disc = std::sqrt(discriminant);
			const double t1 = (-b_eq - sqrt_disc) / (2.0 * a_eq);
			const double t2 = (-b_eq + sqrt_disc) / (2.0 * a_eq);
			
			// Prendi la soluzione positiva più piccola
			double t = std::numeric_limits<double>::max();
			if (t1 > EPS) t = std::min(t, t1);
			if (t2 > EPS) t = std::min(t, t2);
			
			if (t > std::abs(v_dot_d)) {
				return std::abs(v_dot_d); // Non andare oltre il piano
			}
			
			return t;
		};
		// Filter points to keep only those without other points in their cone.
		std::vector<uint> filterIndices;
		filterIndices.reserve(clampedCells.size());
		for (uint i = 0; i < clampedCells.size(); ++i) {
			if (clampedCells[i].distance < MAX_DISTANCE && clampedCells[i].hasHit) {
				filterIndices.push_back(i);
			}
		}

		if (debug) {
			std::cout << "Beginning Clamping phase...\n";
			std::cout.flush();
		}

		auto computeClampedCell = [&](uint i) -> CellData {
			const CellData baseCell = clampedCells[i];
			/*if (!baseCell.hasHit) {
				return baseCell;
			}*/

			const Point3d original = baseCell.hitPoint;
			double t_required = 0.0;

			

			bool anyCone = false;

			for (uint j = 0; j < cells.size(); ++j) {
				//if (i == j || !cells[j].hasHit) continue;
				if (i == j) continue;

				if (!isWithinPlaneAngle(original, cells[j].hitPoint, CONE_COS_THRESHOLD))
					continue;
				
				anyCone = true;

				const double t = coneBoundaryStep(original, cells[j].hitPoint);
				if (t > t_required) {
					t_required = t;
				}
			}

			if (!anyCone) {
				return baseCell;
			}

			// Sposta il punto verso il piano
			const Point3d currentPoint = original - direction * t_required;
			const double distanceToPlane = std::abs((currentPoint - planePoint).dot(direction));
			
			return CellData{distanceToPlane, currentPoint, true};
		};

		/*for (uint idx = 0; idx < filterIndices.size(); ++idx) {
			if (idx % 1000 == 0) {
				std::cout << "Filtering progress: " << idx
						  << "/" << filterIndices.size() << "\n";
				std::cout.flush();
			}
			const uint i = filterIndices[idx];
			clampedCells[i] = computeClampedCell(idx);
		}*/

		vcl::parallelFor(allCells, [&](uint idx) {
			clampedCells[idx] = computeClampedCell(idx);
		});


		auto validateClampedCells = [&]() {
			std::atomic<uint> violatingPoints{0};
			
			vcl::parallelFor(allCells, [&](uint i) {
				if (!clampedCells[i].hasHit) return;

				const Point3d& point = clampedCells[i].hitPoint;

				for (uint j = 0; j < clampedCells.size(); ++j) {
					if (i == j || !clampedCells[j].hasHit) continue;

					const Point3d dirToOther = (clampedCells[j].hitPoint - point);
					const double norm = dirToOther.norm();
					if (norm < EPS) continue;
					
					const double cosVal = (dirToOther / norm).dot(-direction);

					if (cosVal > CONE_COS_THRESHOLD + EPS) {
						violatingPoints.fetch_add(1);
						return;
					}
				}
			});

			uint totalViolations = violatingPoints.load();
			std::cout << "Clamped validation: "
					<< (totalViolations == 0 ? "OK" : "VIOLATIONS")
					<< " (violating points: " << totalViolations << ")\n";
			std::cout.flush();
		};
		


		if (debug) {
			std::cout << "Validating clamped cells...\n";
			std::cout.flush();

			validateClampedCells();
			vcl::PolyMesh hitPointsMesh;
			for (uint i = 0; i < cells.size(); ++i) {
				if (cells[i].hasHit) {
					hitPointsMesh.addVertex(cells[i].hitPoint);
				}
			}

			vcl::PolyMesh clampedonlyPointsMesh;
			clampedonlyPointsMesh.enablePerVertexColor();
			for (uint i = 0; i < clampedCells.size(); ++i) {
				if (!clampedCells[i].hasHit) continue;
				if (cells[i].distance == clampedCells[i].distance) continue;
				const uint vId = clampedonlyPointsMesh.addVertex(clampedCells[i].hitPoint);
				clampedonlyPointsMesh.vertex(vId).color() = vcl::Color::Red;
			}

			vcl::PolyMesh clampedPointsMesh;
			clampedPointsMesh.enablePerVertexColor();
			for (uint i = 0; i < clampedCells.size(); ++i) {
				if (!clampedCells[i].hasHit) continue;
				const uint vId = clampedPointsMesh.addVertex(clampedCells[i].hitPoint);
				clampedPointsMesh.vertex(vId).color() = vcl::Color::Blue;
			}

			vcl::PolyMesh missedPointsMesh;
			missedPointsMesh.enablePerVertexColor();
			for (uint i = 0; i < clampedCells.size(); ++i) {
				if (clampedCells[i].hasHit) continue;
				const uint vId = missedPointsMesh.addVertex(clampedCells[i].hitPoint);
				missedPointsMesh.vertex(vId).color() = vcl::Color::Green;
			}

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

			const std::string base = std::string(VCLIB_EXTERNAL_RESULTS_PATH) + "/888_mold_check";
			saveMesh(hitPointsMesh, base + "_hit_points.ply");
			saveMesh(clampedonlyPointsMesh, base + "_clamped_only_points.ply");
			saveMesh(clampedPointsMesh, base + "_all_clamped_points.ply");
			saveMesh(planeMesh, base + "_plane.ply");
			saveMesh(missedPointsMesh, base + "_missed_points.ply");

			std::cout << "Clamped points: " << clampedPointsMesh.vertexCount() << "\n";
			std::cout << "Saved debug meshes:\n"
					<< " - " << base << "_hit_points.ply\n"
					<< " - " << base << "_clamped_only_points.ply\n"
					<< " - " << base << "_all_clamped_points.ply\n"
					<< " - " << base << "_plane.ply\n"
					<< " - " << base << "_missed_points.ply\n";
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
