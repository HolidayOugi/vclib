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

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

int main()
{
	using namespace vcl;

	struct GridChoice
	{
		uint rows = 1;
		uint cols = 1;
		double sideU = 0.0;
		double sideV = 0.0;
	};

	auto chooseGrid = [&](double lenU, double lenV, uint targetCells) -> GridChoice {
		const double minSideFraction = 0.99;

		if (lenU <= 0.0 || lenV <= 0.0 || targetCells == 0) {
			return {1, 1, lenU, lenV};
		}

		GridChoice best;
		bool       foundValid = false;
		double     bestScore  = std::numeric_limits<double>::infinity();

		const uint maxColsToTry = std::max<uint>(200, targetCells * 2 + 1);

		for (uint cols = 1; cols <= maxColsToTry; ++cols) {
			const double rowsIdeal = static_cast<double>(targetCells) / cols;
			if (!std::isfinite(rowsIdeal)) {
				continue;
			}

			const uint rowCandidates[2] = {
				static_cast<uint>(std::max(1.0, std::floor(rowsIdeal))),
				static_cast<uint>(std::max(1.0, std::ceil(rowsIdeal)))};

			for (uint rows : rowCandidates) {
				const uint chosenCells = rows * cols;
				if (chosenCells == 0) {
					continue;
				}

				const double sideU = lenU / cols;
				const double sideV = lenV / rows;
				if (sideU <= 0.0 || sideV <= 0.0) {
					continue;
				}

				const double shorterSide = std::min(sideU, sideV);
				const double longerSide  = std::max(sideU, sideV);
				if (shorterSide / longerSide < minSideFraction) {
					continue;
				}

				const double countPenalty = std::abs(static_cast<double>(chosenCells) - targetCells);
				const double shapePenalty = std::abs(std::log(sideU / sideV));
				const double score        = countPenalty * 1000.0 + shapePenalty;

				if (!foundValid || score < bestScore) {
					foundValid = true;
					bestScore  = score;
					best       = {rows, cols, sideU, sideV};
				}
			}
		}

		if (foundValid) {
			return best;
		}

		// Fallback: one cell if no valid candidate was found in the search window.
		return {1, 1, lenU, lenV};
	};

	const uint targetCells = 10000;

	const Point3d RAW_PLANE_NORMAL = {1.0, 0.4, 0.2};

	PolyMesh m = loadMesh<PolyMesh>(VCLIB_EXAMPLE_MESHES_PATH "/bunny.obj");
	updateBoundingBox(m);

	const double EPS = 1e-6 * m.boundingBox().diagonal();

	Point3d n = RAW_PLANE_NORMAL;
	if (n.norm() <= EPS) {
		std::cerr << "Invalid plane normal.\n";
		return 1;
	}
	n.normalize();

	// Build a support plane: n·x = min_v(n·v), tangent to the mesh.
	double minProj = std::numeric_limits<double>::infinity();
	for (const auto& v : m.vertices()) {
		minProj = std::min(minProj, v.position().dot(n));
	}
	const Point3d planePoint = n * minProj;
	const Planed  plane(planePoint, n);

	Point3d u, v;
	n.orthoBase(u, v);
	if (u.norm() <= EPS || v.norm() <= EPS) {
		std::cerr << "Could not build a stable orthonormal base on the plane.\n";
		return 1;
	}
	u.normalize();
	v.normalize();

	std::vector<Point3d> projectedPoints;
	projectedPoints.reserve(std::distance(m.vertices().begin(), m.vertices().end()));

	double minU = std::numeric_limits<double>::infinity();
	double minV = std::numeric_limits<double>::infinity();
	double maxU = -std::numeric_limits<double>::infinity();
	double maxV = -std::numeric_limits<double>::infinity();

	for (const auto& vert : m.vertices()) {
		const Point3d projected = plane.projectPoint(vert.position());
		projectedPoints.push_back(projected);
		const Point3d rel = projected - planePoint;

		const double pu = rel.dot(u);
		const double pv = rel.dot(v);

		minU = std::min(minU, pu);
		minV = std::min(minV, pv);
		maxU = std::max(maxU, pu);
		maxV = std::max(maxV, pv);
	}

	const std::array<Point3d, 4> bboxCorners = {
		planePoint + u * minU + v * minV,
		planePoint + u * maxU + v * minV,
		planePoint + u * maxU + v * maxV,
		planePoint + u * minU + v * maxV};

	const double lenU = maxU - minU;
	const double lenV = maxV - minV;
	const GridChoice grid = chooseGrid(lenU, lenV, targetCells);

	EdgeMesh projectedPointsMesh;
	projectedPointsMesh.reserveVertices(projectedPoints.size());
	for (const Point3d& p : projectedPoints) {
		projectedPointsMesh.addVertex(p);
	}

	EdgeMesh bbox2dMesh;
	std::array<uint, 4> cornerIds;
	for (uint i = 0; i < bboxCorners.size(); ++i) {
		cornerIds[i] = bbox2dMesh.addVertex(bboxCorners[i]);
	}
	bbox2dMesh.addEdge(cornerIds[0], cornerIds[1]);
	bbox2dMesh.addEdge(cornerIds[1], cornerIds[2]);
	bbox2dMesh.addEdge(cornerIds[2], cornerIds[3]);
	bbox2dMesh.addEdge(cornerIds[3], cornerIds[0]);

	EdgeMesh grid2dMesh;
	for (uint i = 0; i <= grid.cols; ++i) {
		const double cu = minU + (lenU * i) / grid.cols;
		const uint a = grid2dMesh.addVertex(planePoint + u * cu + v * minV);
		const uint b = grid2dMesh.addVertex(planePoint + u * cu + v * maxV);
		grid2dMesh.addEdge(a, b);
	}
	for (uint j = 0; j <= grid.rows; ++j) {
		const double cv = minV + (lenV * j) / grid.rows;
		const uint a = grid2dMesh.addVertex(planePoint + u * minU + v * cv);
		const uint b = grid2dMesh.addVertex(planePoint + u * maxU + v * cv);
		grid2dMesh.addEdge(a, b);
	}

	// Ray tracing: shoot rays from grid cell centers through the mesh.
	embree::Scene scene(m);

	std::vector<std::vector<uint>> faceTriangulations;
	for (const auto& face : m.faces()) {
		const uint faceId = face.index();
		if (faceId >= faceTriangulations.size()) {
			faceTriangulations.resize(faceId + 1);
		}
		faceTriangulations[faceId] = earCut(face);
	}

	struct RaySegment
	{
		Point3d start;
		Point3d end;
		double  length = 0.0;
		double  volume = 0.0;
	};

	std::vector<RaySegment> raySegments;
	double                  totalVolume = 0.0;

	auto addQuadPrism = [](TriMesh& tm,
							const std::array<Point3d, 4>& baseCorners,
						double startOffset,
						double endOffset,
						const Point3d& dir) {
		std::array<Point3d, 4> b;
		std::array<Point3d, 4> t;
		for (uint k = 0; k < 4; ++k) {
			b[k] = baseCorners[k] + dir * startOffset;
			t[k] = baseCorners[k] + dir * endOffset;
		}

		std::array<uint, 8> ids;
		for (uint k = 0; k < 4; ++k) {
			ids[k + 0] = tm.addVertex(b[k]);
			ids[k + 4] = tm.addVertex(t[k]);
		}

		// Bottom and top quads.
		tm.addFace(ids[0], ids[1], ids[2]);
		tm.addFace(ids[0], ids[2], ids[3]);
		tm.addFace(ids[4], ids[6], ids[5]);
		tm.addFace(ids[4], ids[7], ids[6]);

		// Side quads (as two triangles each).
		tm.addFace(ids[0], ids[4], ids[5]);
		tm.addFace(ids[0], ids[5], ids[1]);

		tm.addFace(ids[1], ids[5], ids[6]);
		tm.addFace(ids[1], ids[6], ids[2]);

		tm.addFace(ids[2], ids[6], ids[7]);
		tm.addFace(ids[2], ids[7], ids[3]);

		tm.addFace(ids[3], ids[7], ids[4]);
		tm.addFace(ids[3], ids[4], ids[0]);
	};

	TriMesh prismsMesh;

	for (uint j = 0; j < grid.rows; ++j) {
		for (uint i = 0; i < grid.cols; ++i) {
			const double cellDu = lenU / grid.cols;
			const double cellDv = lenV / grid.rows;
			const double cellArea = cellDu * cellDv;

			const double u0 = minU + i * cellDu;
			const double u1 = u0 + cellDu;
			const double v0 = minV + j * cellDv;
			const double v1 = v0 + cellDv;

			const std::array<Point3d, 4> cellCorners = {
				planePoint + u * u0 + v * v0,
				planePoint + u * u1 + v * v0,
				planePoint + u * u1 + v * v1,
				planePoint + u * u0 + v * v1};

			const double centerU = minU + (i + 0.5) * (lenU / grid.cols);
			const double centerV = minV + (j + 0.5) * (lenV / grid.rows);
			const Point3d cellCenter = planePoint + u * centerU + v * centerV;

			// Shoot ray in the direction of the supporting plane normal.
			float nearDist = EPS;
			std::vector<Point3d> hitPoints;

			while (true) { 
				std::vector<Point3d>              origins = {cellCenter};
				std::vector<Point3d>              directions = {n};
				std::vector<embree::Scene::HitResult> hits =
					scene.firstFaceIntersectedByRays(origins, directions, nearDist);

				auto [hitFaceId, barCoords, hitTriId] = hits[0];

				if (hitFaceId == UINT_NULL) {
					// No hit in this phase.
					break;
				}

				if (hitFaceId >= faceTriangulations.size()) {
					break;
				}

				const auto& face    = m.face(hitFaceId);
				const auto& hitTris = faceTriangulations[hitFaceId];
				const uint  base    = hitTriId * 3;
				if (base + 2 >= hitTris.size()) {
					break;
				}

				const Point3d& q0 = face.vertex(hitTris[base + 0])->position();
				const Point3d& q1 = face.vertex(hitTris[base + 1])->position();
				const Point3d& q2 = face.vertex(hitTris[base + 2])->position();

				const Point3d hitPoint =
					q0 * barCoords.x() + q1 * barCoords.y() + q2 * barCoords.z();

				hitPoints.push_back(hitPoint);

				// Update near distance to skip past this hit point.
				nearDist =
					static_cast<float>((hitPoint - cellCenter).norm() + 100.0 * EPS);
			}

			if (!hitPoints.empty()) {
				const Point3d segStart = cellCenter;
				const Point3d segEnd   = hitPoints[0];
				const double  startD   = 0.0;
				const double  endD     = (segEnd - cellCenter).dot(n);
				if (endD > startD + EPS) {
					const double segLength = endD - startD;
					const double segVolume = cellArea * segLength;
					raySegments.push_back({segStart, segEnd, segLength, segVolume});
					totalVolume += segVolume;
					addQuadPrism(prismsMesh, cellCorners, startD, endD, n);
				}
			}

			for (uint h = 1; h + 1 < hitPoints.size(); h += 2) {
				const Point3d segStart = hitPoints[h];
				const Point3d segEnd   = hitPoints[h + 1];
				const double  startD   = (segStart - cellCenter).dot(n);
				const double  endD     = (segEnd - cellCenter).dot(n);
				if (endD <= startD + EPS) {
					continue;
				}

				const double segLength = endD - startD;
				const double segVolume = cellArea * segLength;
				raySegments.push_back({segStart, segEnd, segLength, segVolume});
				totalVolume += segVolume;
				addQuadPrism(prismsMesh, cellCorners, startD, endD, n);
			}
		}
	}

	// Build ray mesh.
	EdgeMesh rayhitMesh;

	for (const auto& rs : raySegments) {
		uint va = rayhitMesh.addVertex(rs.start);
		uint vb = rayhitMesh.addVertex(rs.end);
		rayhitMesh.addEdge(va, vb);
	}

	saveMesh(rayhitMesh, VCLIB_RESULTS_PATH "/777_plane_beam_rayhits.ply");
	saveMesh(prismsMesh, VCLIB_RESULTS_PATH "/777_plane_beam_prisms.ply");
	saveMesh(projectedPointsMesh, VCLIB_RESULTS_PATH "/777_plane_beam_projected_points.ply");
	saveMesh(bbox2dMesh, VCLIB_RESULTS_PATH "/777_plane_beam_bbox2d.ply");
	saveMesh(grid2dMesh, VCLIB_RESULTS_PATH "/777_plane_beam_grid2d.ply");

	std::cout << "Projected points: " << projectedPoints.size() << "\n"
			  << "Plane normal: " << n << "\n"
			  << "Target cells: " << targetCells << "\n"
			  << "Chosen grid: " << grid.cols << " x " << grid.rows
			  << " (" << grid.cols * grid.rows << " cells)\n"
			  << "Cell side lengths: " << grid.sideU << " x " << grid.sideV
			  << "\n"
			  << "Prism segments (alternate-hit rule): " << raySegments.size() << "\n"
			  << "Total volume: " << totalVolume << "\n"
			  << "2D bounding box (u,v):\n"
			  << "  min = (" << minU << ", " << minV << ")\n"
			  << "  max = (" << maxU << ", " << maxV << ")\n"
			  << "Saved debug meshes:\n"
			  << " - " << VCLIB_RESULTS_PATH << "/777_plane_beam_prisms.ply\n"
			  << " - " << VCLIB_RESULTS_PATH << "/777_plane_beam_projected_points.ply\n"
			  << " - " << VCLIB_RESULTS_PATH << "/777_plane_beam_bbox2d.ply\n"
			  << " - " << VCLIB_RESULTS_PATH << "/777_plane_beam_grid2d.ply\n"
		      << " - " << VCLIB_RESULTS_PATH << "/777_plane_beam_rayhits.ply\n";

    return 0;
}