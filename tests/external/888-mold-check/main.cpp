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
#include <vclib/algorithms/core/fibonacci.h>



#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>

//STRUCTS

struct GridChoice
{
	vcl::uint rows = 1;
	vcl::uint cols = 1;

	double sideU = 0.0;
	double sideV = 0.0;

	double minU = 0.0;
	double minV = 0.0;
	double maxU = 0.0;
	double maxV = 0.0;
};

struct CellData
	{
	std::array<vcl::Point3d, 4> cellCorners;
	vcl::Point3d cellCenter;
	double distance;
	vcl::Point3d hitPoint;
	bool hasHit = false;
	};

struct ConnectedComponentData
{
	std::vector<vcl::uint> indices;
	double area = 0.0;
	double perimeter = 0.0;
	double compactness = 0.0;
};

//MAKE PLANE
static std::tuple<vcl::Point3d, vcl::Point3d> makePlane(
	const vcl::PolyMesh& m,
	const vcl::Planed&   plane,
	const vcl::Point3d&  planePoint,
	const vcl::Point3d&  direction,
	double               margin,
	double               eps,
	GridChoice&          grid)
{
	using namespace vcl;

	Point3d u;
	Point3d v;

	direction.orthoBase(u, v);

	if (u.norm() <= eps || v.norm() <= eps) {
		grid.minU = 0.0;
		grid.minV = 0.0;
		grid.maxU = 0.0;
		grid.maxV = 0.0;
		return {u, v};
	}

	u.normalize();
	v.normalize();

	grid.minU = std::numeric_limits<double>::infinity();
	grid.minV = std::numeric_limits<double>::infinity();
	grid.maxU = -std::numeric_limits<double>::infinity();
	grid.maxV = -std::numeric_limits<double>::infinity();

	for (const auto& vert : m.vertices()) {
		const Point3d projected = plane.projectPoint(vert.position());
		const Point3d rel       = projected - planePoint;

		const double pu = rel.dot(u);
		const double pv = rel.dot(v);

		grid.minU = std::min(grid.minU, pu - margin);
		grid.minV = std::min(grid.minV, pv - margin);
		grid.maxU = std::max(grid.maxU, pu + margin);
		grid.maxV = std::max(grid.maxV, pv + margin);
	}

	return {u, v};
}

// MAKE GRID
static void makeGrid(
	GridChoice&                grid,
	const std::vector<double>& gridCellSideLengths)
{

	using namespace vcl;

	const double lenU = grid.maxU - grid.minU;
	const double lenV = grid.maxV - grid.minV;

	const double sideU =
		(gridCellSideLengths.size() >= 1) ? gridCellSideLengths[0] : lenU;

	const double sideV =
		(gridCellSideLengths.size() >= 2) ? gridCellSideLengths[1] : sideU;

	if (sideU <= 0.0 || sideV <= 0.0) {
		grid.rows = 1;
		grid.cols = 1;
		grid.sideU = lenU;
		grid.sideV = lenV;
		return;
	}

	grid.cols = static_cast<uint>(std::max(1.0, std::ceil(lenU / sideU)));
	grid.rows = static_cast<uint>(std::max(1.0, std::ceil(lenV / sideV)));

	grid.sideU = sideU;
	grid.sideV = sideV;

	grid.maxU = grid.minU + grid.cols * grid.sideU;
	grid.maxV = grid.minV + grid.rows * grid.sideV;
}

// SHOOT RAY FROM CELL CENTER

static CellData shootRayOnCell(
	const CellData& cell,
	const vcl::PolyMesh& m,
	const vcl::embree::Scene& scene,
	const vcl::Point3d& planePoint,
	const vcl::Point3d& direction,
	double maxDistance,
	double eps)
{
	using namespace vcl;

	const Point3d rayOrigin =
		cell.cellCenter + direction * (-eps);

	const Point3d invalidPoint =
		cell.cellCenter + direction * maxDistance;

	auto [faceId, baryCoords, triId] =
		scene.firstFaceIntersectedByRay(rayOrigin, direction);

	if (faceId != UINT_NULL) {
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
				p0 * baryCoords.x() +
				p1 * baryCoords.y() +
				p2 * baryCoords.z();

			const double distance =
				std::abs((hitPoint - planePoint).dot(direction));

			CellData result = cell;
			result.distance = distance;
			result.hitPoint = hitPoint;
			result.hasHit = true;

			return result;
		}
	}

	CellData result = cell;
	result.distance = maxDistance;
	result.hitPoint = invalidPoint;
	result.hasHit = false;

	return result;
}

// MAKE CELL

static CellData makeCellGeometry(
	vcl::uint idx,
	const GridChoice& grid,
	const vcl::Point3d& planePoint,
	const vcl::Point3d& u,
	const vcl::Point3d& v)
{
	using namespace vcl;

	const uint j = idx / grid.cols;
	const uint i = idx % grid.cols;

	const double u0 = grid.minU + i * grid.sideU;
	const double u1 = u0 + grid.sideU;

	const double v0 = grid.minV + j * grid.sideV;
	const double v1 = v0 + grid.sideV;

	CellData cell;

	cell.cellCorners = {
		planePoint + u * u0 + v * v0,
		planePoint + u * u1 + v * v0,
		planePoint + u * u1 + v * v1,
		planePoint + u * u0 + v * v1};

	const double centerU = grid.minU + (i + 0.5) * grid.sideU;
	const double centerV = grid.minV + (j + 0.5) * grid.sideV;

	cell.cellCenter = planePoint + u * centerU + v * centerV;

	cell.distance = 0.0;
	cell.hitPoint = cell.cellCenter;
	cell.hasHit = false;

	return cell;
}

// CLAMP CELL RELATED FUNCTIONS

static bool isWithinPlaneAngle(
	const vcl::Point3d& point,
	const vcl::Point3d& other,
	const vcl::Point3d& direction,
	double coneCosThreshold,
	double eps)
{	

	using namespace vcl;

	const Point3d directionToOther = other - point;
	const double directionToOtherNorm = directionToOther.norm();

	if (directionToOtherNorm < eps) {
		return true;
	}

	const Point3d dirToPlane = -direction;

	const Point3d dirToOtherNormalized =
		directionToOther / directionToOtherNorm;

	const double cosBetween =
		dirToOtherNormalized.dot(dirToPlane);

	return cosBetween > coneCosThreshold - eps;
}

static double coneBoundaryStep(
	const vcl::Point3d& a,
	const vcl::Point3d& b,
	const vcl::Point3d& direction,
	double coneCosThreshold,
	double eps)
{

	using namespace vcl;

	const Point3d ab = b - a;
	const double abNorm = ab.norm();

	if (abNorm < eps) {
		return 0.0;
	}

	// If b is already outside the cone of a, no displacement is needed.
	// The cone has its apex at a and opens toward the plane, i.e. along -direction.
	const double cosAngle = (ab / abNorm).dot(-direction);

	if (cosAngle <= coneCosThreshold) {
		return 0.0;
	}

	// We want to move a toward the plane:
	//
	//     newA = a - t * direction
	//
	// such that b lies exactly on the cone boundary of newA.
	//
	// The cone boundary condition is:
	//
	//     dot((b - newA) / |b - newA|, -direction) = coneCosThreshold
	//
	// Since:
	//
	//     b - newA = b - (a - t * direction)
	//              = (b - a) + t * direction
	//
	// let:
	//
	//     v = b - a
	//     d = direction
	//
	// so the displaced vector becomes:
	//
	//     v(t) = v + t * d
	//
	// We solve:
	//
	//     dot(v(t) / |v(t)|, -d) = coneCosThreshold
	//
	// which expands to:
	//
	//     -(dot(v, d) + t) / sqrt(|v|^2 + 2t dot(v, d) + t^2)
	//         = coneCosThreshold
	//
	// Squaring both sides gives a quadratic equation in t.

	const Point3d vec = b - a;

	const double vecDotDirection = vec.dot(direction);
	const double vecNorm2 = vec.dot(vec);

	const double cos2 = coneCosThreshold * coneCosThreshold;
	const double sin2 = 1.0 - cos2;

	// Quadratic equation:
	//
	//     sin^2(theta) * t^2
	//   + 2 * dot(v, d) * sin^2(theta) * t
	//   + dot(v, d)^2 - cos^2(theta) * |v|^2
	//   = 0
	//
	// where cos(theta) = coneCosThreshold.

	const double aEq = sin2;
	const double bEq = 2.0 * vecDotDirection * sin2;
	const double cEq =
		vecDotDirection * vecDotDirection -
		cos2 * vecNorm2;

	const double discriminant =
		bEq * bEq - 4.0 * aEq * cEq;

	if (discriminant <= 0.0) {
		// No valid real solution was found.
		// Fall back to the maximum safe displacement along the direction axis.
		return std::abs(vecDotDirection);
	}

	const double sqrtDisc = std::sqrt(discriminant);

	const double t1 =
		(-bEq - sqrtDisc) / (2.0 * aEq);

	const double t2 =
		(-bEq + sqrtDisc) / (2.0 * aEq);

	// Pick the smallest positive displacement.
	double t = std::numeric_limits<double>::max();

	if (t1 > eps) {
		t = std::min(t, t1);
	}

	if (t2 > eps) {
		t = std::min(t, t2);
	}

	if (t == std::numeric_limits<double>::max()) {
		return 0.0;
	}

	// Do not move farther than the projection of b - a along direction.
	// This prevents overshooting past the plane-aligned limit.
	if (t > std::abs(vecDotDirection)) {
		return std::abs(vecDotDirection);
	}

	return t;
}

static CellData computeClampedCell(
	vcl::uint i,
	const std::vector<CellData>& cells,
	const vcl::Point3d& planePoint,
	const vcl::Point3d& direction,
	double coneCosThreshold,
	double eps)
{
	using namespace vcl;

	const CellData baseCell = cells[i];

	const vcl::Point3d original = baseCell.hitPoint;

	double requiredT = 0.0;
	bool anyCone = false;

	for (uint j = 0; j < cells.size(); ++j) {
		if (i == j || !cells[j].hasHit) {
			continue;
		}

		if (!isWithinPlaneAngle(
				original,
				cells[j].hitPoint,
				direction,
				coneCosThreshold,
				eps)) {
			continue;
		}

		anyCone = true;

		const double t = coneBoundaryStep(
			original,
			cells[j].hitPoint,
			direction,
			coneCosThreshold,
			eps);

		requiredT = std::max(requiredT, t);
	}

	if (!anyCone) {
		return baseCell;
	}

	const vcl::Point3d currentPoint =
		original - direction * requiredT;

	const double distanceToPlane =
		std::abs((currentPoint - planePoint).dot(direction));

	return CellData{
		baseCell.cellCorners,
		baseCell.cellCenter,
		distanceToPlane,
		currentPoint,
		true};
}

//VALIDATION

static void validateClampedCells(
	const std::vector<CellData>& clampedCells,
	const std::vector<vcl::uint>& allCells,
	const vcl::Point3d& direction,
	double coneCosThreshold,
	double eps)
{

	using namespace vcl;

	std::atomic<uint> violatingPoints{0};

	vcl::parallelFor(allCells, [&](uint i) {
		if (!clampedCells[i].hasHit) {
			return;
		}

		const vcl::Point3d& point = clampedCells[i].hitPoint;

		for (uint j = 0; j < clampedCells.size(); ++j) {
			if (i == j || !clampedCells[j].hasHit) {
				continue;
			}

			const vcl::Point3d dirToOther =
				clampedCells[j].hitPoint - point;

			const double norm = dirToOther.norm();

			if (norm < eps) {
				continue;
			}

			const double cosVal =
				(dirToOther / norm).dot(-direction);

			if (cosVal > coneCosThreshold + eps) {
				violatingPoints.fetch_add(1);
				return;
			}
		}
	});

	const uint totalViolations = violatingPoints.load();

	std::cout << "Clamped validation: "
			  << (totalViolations == 0 ? "OK" : "VIOLATIONS")
			  << " (violating points: " << totalViolations << ")\n";

	std::cout.flush();
}

static bool isSameDistanceCell(
	const std::vector<CellData>& cells,
	const std::vector<CellData>& clampedCells,
	vcl::uint idx,
	double eps)
{
	return cells[idx].hasHit &&
		   clampedCells[idx].hasHit &&
		   std::abs(cells[idx].distance - clampedCells[idx].distance) <= eps;
}

static void pushNeighbor(
	std::vector<vcl::uint>& stack,
	std::vector<bool>& visited,
	const std::vector<CellData>& cells,
	const std::vector<CellData>& clampedCells,
	vcl::uint neighbor,
	double eps)
{
	if (!visited[neighbor] &&
		isSameDistanceCell(cells, clampedCells, neighbor, eps)) {
		visited[neighbor] = true;
		stack.push_back(neighbor);
	}
}

static double componentGridPerimeter(
	const std::vector<vcl::uint>& componentIndices,
	const GridChoice& grid)
{
	using namespace vcl;

	double perimeter = 0.0;
	std::unordered_set<uint> componentSet(
		componentIndices.begin(), componentIndices.end());

	for (uint idx : componentIndices) {
		const uint row = idx / grid.cols;
		const uint col = idx % grid.cols;

		if (col == 0 || componentSet.count(idx - 1) == 0) {
			perimeter += grid.sideV;
		}
		if (col + 1 == grid.cols || componentSet.count(idx + 1) == 0) {
			perimeter += grid.sideV;
		}
		if (row == 0 || componentSet.count(idx - grid.cols) == 0) {
			perimeter += grid.sideU;
		}
		if (row + 1 == grid.rows || componentSet.count(idx + grid.cols) == 0) {
			perimeter += grid.sideU;
		}
	}

	return perimeter;
}

static ConnectedComponentData largestConnectedComponent(
	const std::vector<CellData>& cells,
	const std::vector<CellData>& clampedCells,
	const GridChoice& grid,
	double eps)
{
	using namespace vcl;

	ConnectedComponentData result;

	if (cells.size() != clampedCells.size() ||
		cells.size() != grid.rows * grid.cols) {
		return result;
	}

	std::vector<bool> visited(cells.size(), false);

	//BFS

	for (uint start = 0; start < cells.size(); ++start) {
		if (visited[start] ||
			!isSameDistanceCell(cells, clampedCells, start, eps)) {
			continue;
		}

		std::vector<uint> component;
		std::vector<uint> stack;

		visited[start] = true;
		stack.push_back(start);

		while (!stack.empty()) {
			const uint idx = stack.back();
			stack.pop_back();
			component.push_back(idx);

			const uint row = idx / grid.cols;
			const uint col = idx % grid.cols;

			if (col > 0) {
				const uint neighbor = idx - 1;
				pushNeighbor(
					stack, visited, cells, clampedCells, neighbor, eps);
			}
			if (col + 1 < grid.cols) {
				const uint neighbor = idx + 1;
				pushNeighbor(
					stack, visited, cells, clampedCells, neighbor, eps);
			}
			if (row > 0) {
				const uint neighbor = idx - grid.cols;
				pushNeighbor(
					stack, visited, cells, clampedCells, neighbor, eps);
			}
			if (row + 1 < grid.rows) {
				const uint neighbor = idx + grid.cols;
				pushNeighbor(
					stack, visited, cells, clampedCells, neighbor, eps);
			}
		}

		if (component.size() > result.indices.size()) {
			result.indices = std::move(component);
		}
	}

	for (uint idx : result.indices) {
		result.area += grid.sideU * grid.sideV;
	}

	result.perimeter = componentGridPerimeter(result.indices, grid);
	result.compactness = (result.perimeter > 0.0) ? (result.area / result.perimeter) : 0.0;

	return result;
}

static vcl::TriMesh createMoldSurface(
	const std::vector<CellData>& clampedCells,
	const GridChoice& grid,
	const vcl::Point3d& direction)
{
	using namespace vcl;

	vcl::TriMesh tm;

	for (uint row = 0; row + 1 < grid.rows; row += 1) {
		for (uint col = 0; col + 1 < grid.cols; col += 1) {
			const uint c00 = row * grid.cols + col;
			const uint c10 = c00 + 1;
			const uint c01 = c00 + grid.cols;
			const uint c11 = c01 + 1;

			const std::array<const CellData*, 4> cells = {
				&clampedCells[c00],
				&clampedCells[c10],
				&clampedCells[c01],
				&clampedCells[c11]};

			

			const double averageDistance =
				(cells[0]->distance +
				 cells[1]->distance +
				 cells[2]->distance +
				 cells[3]->distance) *
				0.25;

			const Point3d foot =
				(cells[0]->cellCenter +
				 cells[1]->cellCenter +
				 cells[2]->cellCenter +
				 cells[3]->cellCenter) *
				0.25;

			const Point3d medianPoint = foot + direction * averageDistance;

			const Point3d p0 = cells[0]->hitPoint;
			const Point3d p1 = cells[1]->hitPoint;
			const Point3d p2 = cells[2]->hitPoint;
			const Point3d p3 = cells[3]->hitPoint;

			const uint v0 = tm.addVertex(p0);
			const uint v1 = tm.addVertex(p1);
			const uint v2 = tm.addVertex(p2);
			const uint v3 = tm.addVertex(p3);
			const uint vc = tm.addVertex(medianPoint);

			tm.addFace(v0, vc, v1);
			tm.addFace(v1, vc, v3);
			tm.addFace(v3, vc, v2);
			tm.addFace(v2, vc, v0);
		}
	}

	
	

	return tm;
}


//DEBUG PRISM CREATION

static vcl::uint addFaceWithColor(
	vcl::TriMesh& tm,
	vcl::uint          v0,
	vcl::uint          v1,
	vcl::uint          v2,
	const vcl::Color& faceColor)
{
	const vcl::uint fid = tm.addFace(v0, v1, v2);
	tm.face(fid).color() = faceColor;
	return fid;
}

static void addQuadPrism(
	vcl::TriMesh& tm,
	const std::array<vcl::Point3d, 4>& baseCorners,
	double startOffset,
	double endOffset,
	const vcl::Point3d& dir,
	const vcl::Color& faceColor)
{

	using namespace vcl;

	tm.enablePerFaceColor();

	std::array<vcl::Point3d, 4> b;
	std::array<vcl::Point3d, 4> t;

	for (uint k = 0; k < 4; ++k) {
		b[k] = baseCorners[k] + dir * startOffset;
		t[k] = baseCorners[k] + dir * endOffset;
	}

	std::array<uint, 8> ids;

	for (uint k = 0; k < 4; ++k) {
		ids[k + 0] = tm.addVertex(b[k]);
		ids[k + 4] = tm.addVertex(t[k]);
	}

	// Bottom
	addFaceWithColor(tm, ids[0], ids[2], ids[1], faceColor);
	addFaceWithColor(tm, ids[0], ids[3], ids[2], faceColor);

	// Top
	addFaceWithColor(tm, ids[4], ids[5], ids[6], faceColor);
	addFaceWithColor(tm, ids[4], ids[6], ids[7], faceColor);

	// Sides
	addFaceWithColor(tm, ids[0], ids[1], ids[5], faceColor);
	addFaceWithColor(tm, ids[0], ids[5], ids[4], faceColor);

	addFaceWithColor(tm, ids[1], ids[2], ids[6], faceColor);
	addFaceWithColor(tm, ids[1], ids[6], ids[5], faceColor);

	addFaceWithColor(tm, ids[2], ids[3], ids[7], faceColor);
	addFaceWithColor(tm, ids[2], ids[7], ids[6], faceColor);

	addFaceWithColor(tm, ids[3], ids[0], ids[4], faceColor);
	addFaceWithColor(tm, ids[3], ids[4], ids[7], faceColor);
}

//DEBUG SEGMENT CREATION

static void addSegment(
	vcl::EdgeMesh& em,
	const vcl::Point3d& a,
	const vcl::Point3d& b)
{
	const vcl::uint va = em.addVertex(a);
	const vcl::uint vb = em.addVertex(b);

	em.addEdge(va, vb);
}

//DEBUG COLOR POINT CREATION

static void addColoredPoint(
	vcl::PolyMesh& mesh,
	const vcl::Point3d& point,
	const vcl::Color& color)
{
	const vcl::uint vId = mesh.addVertex(point);
	mesh.vertex(vId).color() = color;
}

//DEBUG PLANE CREATION

static vcl::TriMesh makeDebugPlaneMesh(
	const GridChoice& grid,
	const vcl::Point3d& planePoint,
	const vcl::Point3d& u,
	const vcl::Point3d& v)
{
	using namespace vcl;

	TriMesh planeMesh;

	const double lenU = grid.maxU - grid.minU;
	const double lenV = grid.maxV - grid.minV;

	const Point3d p0 =
		planePoint + u * grid.minU + v * grid.minV;

	const Point3d p1 =
		planePoint + u * grid.maxU + v * grid.minV;

	const Point3d p2 =
		planePoint + u * grid.maxU + v * grid.maxV;

	const Point3d p3 =
		planePoint + u * grid.minU + v * grid.maxV;

	const uint v0 = planeMesh.addVertex(p0);
	const uint v1 = planeMesh.addVertex(p1);
	const uint v2 = planeMesh.addVertex(p2);
	const uint v3 = planeMesh.addVertex(p3);

	planeMesh.addFace(v0, v1, v2);
	planeMesh.addFace(v0, v2, v3);

	return planeMesh;
}

vcl::EdgeMesh createPerimeterSegments(
	const std::vector<vcl::uint>& componentIndices,
	const std::vector<CellData>& cells,
	const GridChoice& grid)
{
	using namespace vcl;

	EdgeMesh em;

	std::unordered_set<uint> componentSet(componentIndices.begin(), componentIndices.end());

	for (uint idx : componentIndices) {
		const uint row = idx / grid.cols;
		const uint col = idx % grid.cols;
		const CellData& cell = cells[idx];

		if (col == 0 || componentSet.count(idx - 1) == 0) {
			addSegment(em, cell.cellCorners[0], cell.cellCorners[3]);
		}
		if (col + 1 == grid.cols || componentSet.count(idx + 1) == 0) {
			addSegment(em, cell.cellCorners[1], cell.cellCorners[2]);
		}
		if (row == 0 || componentSet.count(idx - grid.cols) == 0) {
			addSegment(em, cell.cellCorners[0], cell.cellCorners[1]);
		}
		if (row + 1 == grid.rows || componentSet.count(idx + grid.cols) == 0) {
			addSegment(em, cell.cellCorners[3], cell.cellCorners[2]);
		}
	}

	return em;
}

int moldCheck(
	vcl::PolyMesh              m,
	const std::vector<double>& gridCellSideLengths,
	bool                       debug,
	vcl::Point3d 			   direction,
	const double 			   coneAngleDegrees,
	const double 			   marginFactor)
{
	using namespace vcl;

	const double CONE_COS_THRESHOLD = std::cos(coneAngleDegrees * M_PI / 180.0);
    
    if (debug) {
        std::cout << "=== moldCheck started ===\n";
        std::cout.flush();
    }

    updateBoundingBox(m);

	const double MAX_DISTANCE = m.boundingBox().diagonal();
	const double EPS = 1e-12 * MAX_DISTANCE;

    embree::Scene scene(m);

    direction.normalize();

    double minProj = std::numeric_limits<double>::infinity();
		for (const auto& vv : m.vertices()) {
			minProj = std::min(minProj, vv.position().dot(direction));
		}
	
	const Point3d planePoint = direction * minProj;
	const Planed  plane(planePoint, direction);
    const double margin = marginFactor * MAX_DISTANCE;

	GridChoice grid;

	const auto [u, v] = makePlane(
		m,
		plane,
		planePoint,
		direction,
		margin,
		EPS,
		grid);

	const double lenU = grid.maxU - grid.minU;
	const double lenV = grid.maxV - grid.minV;

	if (lenU <= EPS || lenV <= EPS) {
		return 1;
	}

	makeGrid(grid, gridCellSideLengths);

    const double cellDu   = grid.sideU;
	const double cellDv   = grid.sideV;
	const double cellArea = cellDu * cellDv;
	std::vector<uint> allCells(grid.rows * grid.cols);
	std::iota(allCells.begin(), allCells.end(), 0);
	std::vector<CellData> cells(allCells.size());

	parallelFor(allCells, [&](uint idx) {
		const CellData cell = makeCellGeometry(idx, grid, planePoint, u, v);
		cells[idx] = shootRayOnCell(cell, m, scene, planePoint, direction, MAX_DISTANCE, EPS);
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
			std::cout << "Beginning Clamping phase...\n";
			std::cout.flush();
	}


	parallelFor(allCells, [&](uint idx) {
		clampedCells[idx] = computeClampedCell(idx, cells, planePoint, direction, CONE_COS_THRESHOLD, EPS);
	});

	ConnectedComponentData largestComponent =
		largestConnectedComponent(
			cells, clampedCells, grid, EPS);


	if (debug) {
		//std::cout << "Validating clamped cells...\n";
		//std::cout.flush();

		//validateClampedCells(clampedCells, allCells, direction, CONE_COS_THRESHOLD, EPS);
		
		PolyMesh hitPointsMesh;
		hitPointsMesh.enablePerVertexColor();
		for (uint i = 0; i < cells.size(); ++i) {
			if (!cells[i].hasHit) continue;
			addColoredPoint(hitPointsMesh, cells[i].hitPoint, Color::Yellow);
		}
		
		PolyMesh clampedonlyPointsMesh;
		clampedonlyPointsMesh.enablePerVertexColor();
		for (uint i = 0; i < clampedCells.size(); ++i) {
			if (!cells[i].hasHit) continue;
			if (cells[i].distance == clampedCells[i].distance) continue;
			addColoredPoint(clampedonlyPointsMesh, clampedCells[i].hitPoint, Color::Red);
		}
		
		PolyMesh clampednohitPointsMesh;
		clampednohitPointsMesh.enablePerVertexColor();
		for (uint i = 0; i < clampedCells.size(); ++i) {
			if (cells[i].hasHit) continue;
			if (cells[i].distance == clampedCells[i].distance) continue;
			addColoredPoint(clampednohitPointsMesh, clampedCells[i].hitPoint, Color::White);
		}
		
		PolyMesh clampedPointsMesh;
		clampedPointsMesh.enablePerVertexColor();
		for (uint i = 0; i < clampedCells.size(); ++i) {
			if (!clampedCells[i].hasHit) continue;
			addColoredPoint(clampedPointsMesh, clampedCells[i].hitPoint, Color::Blue);
		}
		
		PolyMesh missedPointsMesh;
		missedPointsMesh.enablePerVertexColor();
		for (uint i = 0; i < clampedCells.size(); ++i) {
			if (clampedCells[i].hasHit) continue;
			addColoredPoint(missedPointsMesh, clampedCells[i].hitPoint, Color::Green);
		}

		PolyMesh largestComponentMesh;
		largestComponentMesh.enablePerVertexColor();
		for (uint i : largestComponent.indices) {
			addColoredPoint(largestComponentMesh, clampedCells[i].hitPoint, Color::Cyan);
		}

		const TriMesh planeMesh =
			makeDebugPlaneMesh(grid, planePoint, u, v);

		TriMesh ClampedPrismMesh;
		for (uint i = 0; i < clampedCells.size(); ++i) {
			if (!clampedCells[i].hasHit) continue;
			addQuadPrism(ClampedPrismMesh, clampedCells[i].cellCorners, -EPS, clampedCells[i].distance, direction, vcl::Color::White);
		}
		
		TriMesh remainingMoldMesh;
		EdgeMesh segmentsRemainingMold;
		for (uint i = 0; i < clampedCells.size(); ++i) {
			if (!cells[i].hasHit) continue;
			if (cells[i].distance == clampedCells[i].distance) continue;
			addQuadPrism(remainingMoldMesh, clampedCells[i].cellCorners, clampedCells[i].distance, cells[i].distance, direction, vcl::Color::Red);
			addSegment(segmentsRemainingMold, clampedCells[i].hitPoint, cells[i].hitPoint);
		}

		const TriMesh moldSurfaceMesh = createMoldSurface(clampedCells, grid, direction);

		const EdgeMesh perimeterSegments =
			createPerimeterSegments(
				largestComponent.indices, clampedCells, grid);

		
		const std::string base = std::string(VCLIB_EXTERNAL_RESULTS_PATH) + "/888_mold_check";
		for (const auto& entry : std::filesystem::directory_iterator(VCLIB_EXTERNAL_RESULTS_PATH)) if (entry.is_regular_file() && entry.path().extension() == ".ply") std::filesystem::remove(entry.path());
		saveMesh(hitPointsMesh, base + "_hit_points.ply");
		saveMesh(clampedonlyPointsMesh, base + "_clamped_only_points.ply");
		saveMesh(clampednohitPointsMesh, base + "_clamped_nohit_points.ply");
		saveMesh(clampedPointsMesh, base + "_all_clamped_points.ply");
		saveMesh(planeMesh, base + "_plane.ply");
		saveMesh(missedPointsMesh, base + "_missed_points.ply");
		saveMesh(ClampedPrismMesh, base + "_clamped_prisms.ply");
		saveMesh(remainingMoldMesh, base + "_remaining_mold.ply");
		saveMesh(segmentsRemainingMold, base + "_remaining_mold_segments.ply");
		saveMesh(moldSurfaceMesh, base + "_mold_surface.ply");
		saveMesh(largestComponentMesh, base + "_largest_component_points.ply");
		saveMesh(perimeterSegments, base + "_largest_component_perimeter.ply");
		
		std::cout << "Clamped points: " << clampedPointsMesh.vertexCount() << "\n";
		std::cout << "Mold surface median points: " << moldSurfaceMesh.vertexCount() << "\n";
		std::cout << "Largest component cells: "
				  << largestComponent.indices.size() << "\n";
		std::cout << "Largest component area: "
				  << largestComponent.area << "\n";
		std::cout << "Largest component perimeter: "
				  << largestComponent.perimeter << "\n";
		std::cout << "Largest component compactness: "
				  << largestComponent.compactness << "\n";	
		std::cout << "Saved debug meshes:\n"
				<< " - " << base << "_hit_points.ply\n"
				<< " - " << base << "_clamped_only_points.ply\n"
				<< " - " << base << "_clamped_nohit_points.ply\n"
				<< " - " << base << "_all_clamped_points.ply\n"
				<< " - " << base << "_plane.ply\n"
				<< " - " << base << "_missed_points.ply\n"
				<< " - " << base << "_clamped_prisms.ply\n"
				<< " - " << base << "_remaining_mold.ply\n"
				<< " - " << base << "_remaining_mold_segments.ply\n"
				<< " - " << base << "_mold_surface.ply\n"
				<< " - " << base << "_largest_component_points.ply\n"
				<< " - " << base << "_largest_component_perimeter.ply\n";

		std::cout << "=== moldCheck completed successfully ===\n";
		std::cout.flush();
    }
        
    return largestComponent.indices.size();
}

int main()
{
    using namespace vcl;

	const auto startTime = std::chrono::steady_clock::now();

	const uint NUM_PLANES = 100;

	std::vector<Point3d> fibNormals = sphericalFibonacciPointSet<Point3d>(NUM_PLANES);


    PolyMesh m = loadMesh<PolyMesh>(VCLIB_EXAMPLE_MESHES_PATH "/bimba_enlarged.ply");


    std::vector<double> gridCellSideLengths = {0.4, 0.4};

	const double coneAngleDegrees = 5.0;

	const double marginFactor = 0.1;

	int result = 0;

	int bestResult = 0;
	int bestDirectionIndex = 0;

	for (const auto& direction : fibNormals) {
		std::cout << "Processing direction: " << direction << "\n";
		result = moldCheck(m, gridCellSideLengths, false, direction, coneAngleDegrees, marginFactor);
		if (result < 0) {
			break;
		}
		if (result > bestResult) {
			bestResult = result;
			bestDirectionIndex = &direction - &fibNormals[0];
			std::cout << "New best result: " << bestResult << " (direction index: " << bestDirectionIndex << ")\n";
		}
	}

	result = moldCheck(m, gridCellSideLengths, true, fibNormals[bestDirectionIndex], coneAngleDegrees, marginFactor);

    const auto endTime = std::chrono::steady_clock::now();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "moldCheck execution time: " << elapsedMs.count() << " ms\n";
    std::cout.flush();
    return 0;
}
