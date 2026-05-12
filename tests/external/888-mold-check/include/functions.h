#ifndef VCL_TEST_EXTERNAL_888_MOLD_CHECK_FUNCTIONS_H
#define VCL_TEST_EXTERNAL_888_MOLD_CHECK_FUNCTIONS_H

#include "helper.h"
#include "struct.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#include <vclib/embree/scene.h>
#include <vclib/meshes.h>

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

static CellData shootRayOnCell(
	const CellData& cell,
	const vcl::PolyMesh& m,
	const vcl::embree::Scene& scene,
	const vcl::Point3d& planePoint,
	const vcl::Point3d& direction,
	double maxDistance,
	float eps)
{
	using namespace vcl;

	const Point3d rayOrigin =
		cell.cellCenter + direction * (-eps);

	const Point3d invalidPoint =
		cell.cellCenter + direction * maxDistance;

	//noticeably faster to use firstFaceIntersectedbyRay and then recast multiple
	//rays only on non-empty cells than to use facesIntersectedByRay directly for all cells
	auto [faceId, baryCoords, triId, hitT] =
		scene.firstFaceIntersectedByRay(rayOrigin, direction);

	if (faceId != UINT_NULL) {
		//redoing the first hit might seem redudant but it is actually faster than computing the hit point three times.
		const auto rayHits = scene.facesIntersectedByRay(rayOrigin, direction, eps);

		//fallback for possible missed hit due to numerical issues in firstFaceIntersectedByRay and silent crash
		//possible bug to fix
		if (rayHits.empty()) {
			CellData result = cell;
			result.distance = hitT;
			result.hitPoint = computeHitPoint(m, faceId, triId, baryCoords, invalidPoint);
			result.thirdHitPoint = result.hitPoint; //to remove
			result.hasHit = result.hitPoint != invalidPoint;
			result.hasHiddenHit = false;

			return result;
		}
		
		auto [faceId, baryCoords, triId, hitT] = rayHits.front(); 
		Point3d hitPoint = computeHitPoint(m, faceId, triId, baryCoords, invalidPoint);

		if (hitPoint != invalidPoint) {
			CellData result = cell;
			result.distance = hitT;
			result.hitPoint = hitPoint;
			result.thirdHitPoint = hitPoint; //to remove
			result.hasHit = true;
			result.hasHiddenHit = rayHits.size() > 2;
				
			//to remove
			if (result.hasHiddenHit) {
				auto [thirdFaceId, thirdBaryCoords, thirdTriId, thirdHitT] =
					rayHits[2];

				const Point3d thirdHitPoint =
					computeHitPoint(m, thirdFaceId, thirdTriId, thirdBaryCoords, result.thirdHitPoint);

				result.thirdHitPoint = thirdHitPoint;
			}

			return result;
		}
	}

	CellData result = cell;
	result.distance = maxDistance;
	result.hitPoint = invalidPoint;
	result.thirdHitPoint = invalidPoint;
	result.hasHit = false;
	result.hasHiddenHit = false;

	return result;
}

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
	cell.thirdHitPoint = cell.cellCenter;
	cell.hasHit = false;
	cell.hasHiddenHit = false;

	return cell;
}

static CellData computeClampedCell(
	vcl::uint i,
	const std::vector<CellData>& cells,
	const vcl::Point3d& planePoint,
	const vcl::Point3d& direction,
	double coneCosThreshold,
	float eps)
{
	using namespace vcl;

	const CellData baseCell = cells[i];
	const Point3d original = baseCell.hitPoint;

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

	const Point3d currentPoint =
		original - direction * requiredT;

	const double distanceToPlane =
		std::abs((currentPoint - planePoint).dot(direction));

	return CellData{
		baseCell.cellCorners,
		baseCell.cellCenter,
		distanceToPlane,
		currentPoint,
		baseCell.thirdHitPoint,
		true,
		baseCell.hasHiddenHit};
}

static ConnectedComponentData largestConnectedComponent(
	const std::vector<CellData>& cells,
	const std::vector<CellData>& clampedCells,
	const GridChoice& grid,
	float eps)
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
	result.compactness =
		(result.perimeter > 0.0) ? (result.area / result.perimeter) : 0.0;

	return result;
}

#endif
