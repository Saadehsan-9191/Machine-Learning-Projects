# Pavement Crack Labelling Guide (ASTM D6433 Reference)
**Project:** METI PCI Pipeline (Pavement Crack Detection from UAV Imagery)  
**Author:** Saad Ehsan (ML Intern)

---

## 1. Overview
This guide defines the standard annotation rules and visual criteria for the three core pavement crack classes required for the METI PCI pipeline. All object instances must be annotated with bounding boxes using the `[x_min, y_min, x_max, y_max]` pixel coordinate convention.

---

## 2. Core Detection Classes

### A. Longitudinal Crack (`longitudinal_crack` / RDD2022: D00)
* **Definition:** Cracks that are roughly parallel to the road centerline or the direction of traffic travel.
* **Characteristics:** 
  * Typically formed due to poor construction lane joints or traffic load fatigue.
  * Extends linearly along the long axis of the pavement.
* **Labelling Rule:** Draw a tight bounding box enclosing the continuous linear path of the longitudinal fracture.

### B. Transverse Crack (`transverse_crack` / RDD2022: D10)
* **Definition:** Cracks that run roughly perpendicular (across) to the road centerline or traffic direction.
* **Characteristics:**
  * Usually caused by thermal contraction and expansion of the asphalt surface or sub-base movement.
  * Cuts across travel lanes from side to side.
* **Labelling Rule:** Draw a bounding box framing the cross-wise fracture extending across the pavement surface.

### C. Alligator Crack (`alligator_crack` / RDD2022: D20)
* **Definition:** A series of interconnected cracks forming small, multi-sided polygonal pieces resembling an alligator's skin (fatigue cracking).
* **Characteristics:**
  * Occurs in wheel paths or areas subjected to repeated heavy traffic loading.
  * Appears as a dense web or mesh of intersecting fractures rather than a single linear crack.
* **Labelling Rule:** Draw a bounding box encompassing the entire localized web/network of interconnected fatigue cracks as a single object instance.

---

## 3. Edge Cases & Negative Examples (Do NOT Label)
* **Shadows:** Deep tree or overhead structure shadows cast across the road; ensure they are not confused with linear cracks.
* **Construction Joints:** Clean, factory-sealed pavement joints unless severe secondary cracking radiates outward from them.
* **Surface Markings:** Painted lane lines, arrows, thermoplastic symbols, or rubberized crack-sealing material.