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
    using namespace vcl;

    QApplication app(argc, argv);

    // Runtime/configuration knobs for search, debug geometry and visualization.
    constexpr bool   VISUAL             = true;
    constexpr double EPS                = 1e-9;
    constexpr uint   VIEW_SAMPLE_STRIDE = 10;
    constexpr uint   NUM_PLANES         = 100;
    constexpr bool   VERBOSE_TRIANGLES  = false;

    PolyMesh m = loadMesh<PolyMesh>(VCLIB_EXAMPLE_MESHES_PATH "/greek_helmet.obj");
    embree::Scene scene(m);

    updateBoundingBox(m);

    EdgeMesh bestRaysMesh;
    EdgeMesh bestPrismsMesh;
    EdgeMesh bestRaysMeshView;
    EdgeMesh bestPrismsMeshView;
    TriMesh  bestPlaneMesh;

    auto addSegment = [](EdgeMesh& em, const Point3d& a, const Point3d& b) {
        uint va = em.addVertex(a);
        uint vb = em.addVertex(b);
        em.addEdge(va, vb);
    };

    // Evaluate one candidate plane orientation and return the accumulated volume.
    auto evaluatePlane = [&](const Point3d& rawPlaneNormal,
                             bool           collectDebug,
                             EdgeMesh*      raysMesh,
                             EdgeMesh*      prismsMesh,
                             EdgeMesh*      raysMeshView,
                             EdgeMesh*      prismsMeshView,
                             uint&          outTriCount) -> double {
        Point3d planeNormal = rawPlaneNormal;
        if (planeNormal.norm() <= EPS) {
            outTriCount = 0;
            return std::numeric_limits<double>::infinity();
        }
        planeNormal.normalize();

        // Build a support plane: n·x = min_v(n·v). This keeps the plane tangent
        // to the mesh instead of slicing through it.
        double minProj = std::numeric_limits<double>::infinity();
        for (const auto& v : m.vertices()) {
            minProj = std::min(minProj, v.position().dot(planeNormal));
        }
        const Point3d planePoint = planeNormal * minProj;
        const Planed  plane(planePoint, planeNormal);

        uint   localTriId  = 0;
        double localVolume = 0.0;

        for (const auto& face : m.faces()) {
            std::vector<uint> triangulation = earCut(face);
            if (triangulation.size() < 3) {
                continue;
            }

            for (uint i = 0; i + 2 < triangulation.size(); i += 3) {
                const Point3d& p0 = face.vertex(triangulation[i + 0])->position();
                const Point3d& p1 = face.vertex(triangulation[i + 1])->position();
                const Point3d& p2 = face.vertex(triangulation[i + 2])->position();

                const double area = Triangle<Point3d>::area(p0, p1, p2);
                if (area <= EPS) {
                    continue;
                }

                Point3d triNormalUnit = Triangle<Point3d>::normal(p0, p1, p2);
                if (triNormalUnit.norm() <= EPS) {
                    continue;
                }
                triNormalUnit.normalize();

                const double dot = triNormalUnit.dot(planeNormal);
                // Keep only triangles with opposite orientation w.r.t. plane normal.
                if (dot >= 0.0) {
                    continue;
                }

                const Point3d triCentroid  = (p0 + p1 + p2) / 3.0;
                const Point3d rayDirection = -planeNormal;
                Point3d       rayOrigin    = triCentroid + rayDirection * EPS;

                auto [hitFaceId, barCoords, hitTriId] =
                    scene.firstFaceIntersectedByRay(rayOrigin, rayDirection, EPS);

                if (hitFaceId == face.index()) {
                    rayOrigin = triCentroid + rayDirection * (100.0 * EPS);
                    std::tie(hitFaceId, barCoords, hitTriId) =
                        scene.firstFaceIntersectedByRay(rayOrigin, rayDirection, EPS);
                }

                Point3d targetPoint;
                bool    hitMesh = hitFaceId != UINT_NULL;

                if (hitMesh) {
                    const auto& hitFace = m.face(hitFaceId);
                    auto        hitTris = earCut(hitFace);
                    const uint  base    = hitTriId * 3;

                    if (base + 2 < hitTris.size()) {
                        const Point3d& q0 =
                            hitFace.vertex(hitTris[base + 0])->position();
                        const Point3d& q1 =
                            hitFace.vertex(hitTris[base + 1])->position();
                        const Point3d& q2 =
                            hitFace.vertex(hitTris[base + 2])->position();

                        targetPoint =
                            q0 * barCoords.x() + q1 * barCoords.y() + q2 * barCoords.z();
                    }
                    else {
                        hitMesh = false;
                    }
                }

                if (!hitMesh) {
                    const double denom = planeNormal.dot(rayDirection);
                    if (std::abs(denom) <= EPS) {
                        continue;
                    }

                    const double t =
                        (plane.offset() - planeNormal.dot(rayOrigin)) / denom;
                    if (t < 0.0) {
                        continue;
                    }
                    targetPoint = rayOrigin + rayDirection * t;
                }

                // Prism contribution for this triangle.
                const double height = (targetPoint - triCentroid).norm();
                const double volume = area * -dot * height;
                localVolume += volume;

                if (collectDebug && raysMesh && prismsMesh && raysMeshView &&
                    prismsMeshView) {
                    addSegment(*raysMesh, triCentroid, targetPoint);

                    Point3d b0 = p0 + rayDirection * height;
                    Point3d b1 = p1 + rayDirection * height;
                    Point3d b2 = p2 + rayDirection * height;

                    addSegment(*prismsMesh, p0, p1);
                    addSegment(*prismsMesh, p1, p2);
                    addSegment(*prismsMesh, p2, p0);

                    addSegment(*prismsMesh, b0, b1);
                    addSegment(*prismsMesh, b1, b2);
                    addSegment(*prismsMesh, b2, b0);

                    addSegment(*prismsMesh, p0, b0);
                    addSegment(*prismsMesh, p1, b1);
                    addSegment(*prismsMesh, p2, b2);

                    if (localTriId % VIEW_SAMPLE_STRIDE == 0) {
                        addSegment(*raysMeshView, triCentroid, targetPoint);

                        addSegment(*prismsMeshView, p0, p1);
                        addSegment(*prismsMeshView, p1, p2);
                        addSegment(*prismsMeshView, p2, p0);

                        addSegment(*prismsMeshView, b0, b1);
                        addSegment(*prismsMeshView, b1, b2);
                        addSegment(*prismsMeshView, b2, b0);

                        addSegment(*prismsMeshView, p0, b0);
                        addSegment(*prismsMeshView, p1, b1);
                        addSegment(*prismsMeshView, p2, b2);
                    }

                    if (VERBOSE_TRIANGLES) {
                        std::cout << "  Ray result: "
                                  << (hitMesh ? "hit mesh" : "hit plane")
                                  << ", area = " << area << ", dot = " << dot
                                  << ", height = " << height << ", V = "
                                  << volume << "\n";
                    }
                }

                ++localTriId;
            }
        }

        outTriCount = localTriId;
        std::cout << "Plane normal: " << planeNormal
                  << ", volume: " << localVolume
                  << ", triangles processed: " << localTriId << "\n";
        return localVolume;
    };

    // Generate evenly distributed normals on the sphere (candidate planes).
    std::vector<Point3d> fibNormals = sphericalFibonacciPointSet<Point3d>(NUM_PLANES);
    if (fibNormals.empty()) {
        std::cerr << "No Fibonacci planes generated.\n";
        return 1;
    }

    uint    bestPlaneId    = 0;
    double  bestVolume     = std::numeric_limits<double>::infinity();
    Point3d bestNormal     = fibNormals.front();
    uint    bestTriCount   = 0;

    // First pass: evaluate all planes and keep the one with minimum volume.
    for (uint i = 0; i < fibNormals.size(); ++i) {
        uint   tmpTriCount = 0;
        double vol = evaluatePlane(
            fibNormals[i], false, nullptr, nullptr, nullptr, nullptr, tmpTriCount);

        if (vol < bestVolume) {
            bestVolume   = vol;
            bestPlaneId  = i;
            bestNormal   = fibNormals[i];
            bestTriCount = tmpTriCount;
        }
    }

    // Second pass on the best plane: collect full debug geometry.
    uint debugTriCount = 0;
    bestVolume = evaluatePlane(
        bestNormal,
        true,
        &bestRaysMesh,
        &bestPrismsMesh,
        &bestRaysMeshView,
        &bestPrismsMeshView,
        debugTriCount);

    {
        // Build a finite quad for Qt rendering of the best support plane.
        Point3d n = bestNormal;
        n.normalize();

        double minProj = std::numeric_limits<double>::infinity();
        uint   minVid  = UINT_NULL;
        for (const auto& vtx : m.vertices()) {
            const double proj = vtx.position().dot(n);
            if (proj < minProj) {
                minProj = proj;
                minVid  = vtx.index();
            }
        }

        Point3d u, v;
        n.orthoBase(u, v);
        if (u.norm() > EPS) {
            u.normalize();
        }
        if (v.norm() > EPS) {
            v.normalize();
        }

        const double halfSize = std::max(0.05, m.boundingBox().diagonal() * 0.75);
        const Point3d c = (minVid != UINT_NULL) ? m.vertex(minVid).position() :
                             (n * minProj);

        const uint v0 = bestPlaneMesh.addVertex(c - u * halfSize - v * halfSize);
        const uint v1 = bestPlaneMesh.addVertex(c + u * halfSize - v * halfSize);
        const uint v2 = bestPlaneMesh.addVertex(c + u * halfSize + v * halfSize);
        const uint v3 = bestPlaneMesh.addVertex(c - u * halfSize + v * halfSize);

        bestPlaneMesh.addFace(v0, v1, v2);
        bestPlaneMesh.addFace(v0, v2, v3);
    }

    vcl::saveMesh(bestRaysMesh, VCLIB_RESULTS_PATH "/666_volume_prism_rays.ply");
    vcl::saveMesh(bestPrismsMesh, VCLIB_RESULTS_PATH "/666_volume_prism_prisms.ply");

    std::cout << "\nFibonacci planes tested: " << fibNormals.size()
              << "\nBest plane id: " << bestPlaneId
              << "\nBest plane normal: " << bestNormal
              << "\nTriangles processed on best plane: " << bestTriCount
              << "\nTriangles processed while collecting debug: " << debugTriCount
              << "\nMinimum volume: " << bestVolume << "\n";
    std::cout << "Saved debug meshes:\n"
              << " - " << VCLIB_RESULTS_PATH << "/666_volume_prism_rays.ply\n"
              << " - " << VCLIB_RESULTS_PATH
              << "/666_volume_prism_prisms.ply\n";

    if (VISUAL) {
        // Show mesh + best plane + sampled debug rays/prisms.
        MeshViewerSelectQt mv;

        DrawableMesh<PolyMesh> drawableInput(std::move(m));
        drawableInput.name() = "input_mesh";
        drawableInput.updateBuffers();

        DrawableMesh<TriMesh> drawablePlane(std::move(bestPlaneMesh));
        drawablePlane.name() = "best_plane";
        drawablePlane.updateBuffers();

        using enum vcl::MeshRenderInfo::Buffers;

        DrawableMesh<EdgeMesh> drawableRays(std::move(bestRaysMeshView));
        drawableRays.name() = "debug_rays";
        drawableRays.updateBuffers({VERTICES, EDGES});

        DrawableMesh<EdgeMesh> drawablePrisms(std::move(bestPrismsMeshView));
        drawablePrisms.name() = "debug_prisms";
        drawablePrisms.updateBuffers({VERTICES, EDGES});

        auto vec = std::make_shared<DrawableObjectVector>();
        vec->pushBack(std::move(drawableInput));
        vec->pushBack(std::move(drawablePlane));
        vec->pushBack(std::move(drawableRays));
        vec->pushBack(std::move(drawablePrisms));

        mv.setDrawableObjectVector(vec);

        mv.show();
        mv.showMaximized();

        return app.exec();
    }

    return 0;
}
