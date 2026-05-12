#ifndef VCL_TEST_EXTERNAL_888_MOLD_CHECK_DEBUG_OUTPUT_H
#define VCL_TEST_EXTERNAL_888_MOLD_CHECK_DEBUG_OUTPUT_H

#include "struct.h"

#include <array>
#include <atomic>
#include <iostream>
#include <unordered_set>
#include <vector>

#include <vclib/meshes.h>

static void validateClampedCells(
	const std::vector<CellData>& clampedCells,
	const std::vector<vcl::uint>& allCells,
	const vcl::Point3d& direction,
	double coneCosThreshold,
	float eps)
{
	using namespace vcl;

	std::atomic<uint> violatingPoints{0};

	vcl::parallelFor(allCells, [&](uint i) {
		if (!clampedCells[i].hasHit) {
			return;
		}

		const Point3d& point = clampedCells[i].hitPoint;

		for (uint j = 0; j < clampedCells.size(); ++j) {
			if (i == j || !clampedCells[j].hasHit) {
				continue;
			}

			const Point3d dirToOther =
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

static vcl::TriMesh createMoldSurface(
	const std::vector<CellData>& clampedCells,
	const GridChoice& grid,
	const vcl::Point3d& direction)
{
	using namespace vcl;

	TriMesh tm;

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

static vcl::uint addFaceWithColor(
	vcl::TriMesh& tm,
	vcl::uint v0,
	vcl::uint v1,
	vcl::uint v2,
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

	addFaceWithColor(tm, ids[0], ids[2], ids[1], faceColor);
	addFaceWithColor(tm, ids[0], ids[3], ids[2], faceColor);

	addFaceWithColor(tm, ids[4], ids[5], ids[6], faceColor);
	addFaceWithColor(tm, ids[4], ids[6], ids[7], faceColor);

	addFaceWithColor(tm, ids[0], ids[1], ids[5], faceColor);
	addFaceWithColor(tm, ids[0], ids[5], ids[4], faceColor);

	addFaceWithColor(tm, ids[1], ids[2], ids[6], faceColor);
	addFaceWithColor(tm, ids[1], ids[6], ids[5], faceColor);

	addFaceWithColor(tm, ids[2], ids[3], ids[7], faceColor);
	addFaceWithColor(tm, ids[2], ids[7], ids[6], faceColor);

	addFaceWithColor(tm, ids[3], ids[0], ids[4], faceColor);
	addFaceWithColor(tm, ids[3], ids[4], ids[7], faceColor);
}

static void addSegment(
	vcl::EdgeMesh& em,
	const vcl::Point3d& a,
	const vcl::Point3d& b)
{
	const vcl::uint va = em.addVertex(a);
	const vcl::uint vb = em.addVertex(b);

	em.addEdge(va, vb);
}

static void addColoredPoint(
	vcl::PolyMesh& mesh,
	const vcl::Point3d& point,
	const vcl::Color& color)
{
	const vcl::uint vId = mesh.addVertex(point);
	mesh.vertex(vId).color() = color;
}

static vcl::TriMesh makeDebugPlaneMesh(
	const GridChoice& grid,
	const vcl::Point3d& planePoint,
	const vcl::Point3d& u,
	const vcl::Point3d& v)
{
	using namespace vcl;

	TriMesh planeMesh;

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

static vcl::EdgeMesh createPerimeterSegments(
	const std::vector<vcl::uint>& componentIndices,
	const std::vector<CellData>& cells,
	const GridChoice& grid)
{
	using namespace vcl;

	EdgeMesh em;

	std::unordered_set<uint> componentSet(
		componentIndices.begin(), componentIndices.end());

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

#endif
