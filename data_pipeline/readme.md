# 📋 Detailed Dataset Collection Process

## 1️⃣ Construction of Modular 3D Block Simulation Assets

![Modular 3D Blocks](https://github.com/PhyBlock/PhyBlock/blob/main/Imgs/PhyBlock_Assets.png)

We utilize Blender as the primary platform to create our simulation assets through a two-stage process:

1. **Geometric Modeling**:
   Parameterized 3D models are constructed in Blender using a base unit of 5 cm. Boolean operations are applied to generate eight standard geometric shapes, ensuring modularity and consistency across assets.

2. **Physical Material Modeling**:
   The geometric models are imported into Blender 3.4 for configuring rigid-body dynamics. To realistically simulate ABS plastic, we set the friction coefficient to μ = 0.35 and the density to ρ = 1.04 g/cm³. Additionally, normal mapping and edge subdivision techniques are applied to enhance surface details and improve edge fidelity.

The final style and scale of the block assets are illustrated in the figure above. The complete set of 3D block assets is publicly available in our repository at [`data/3D-Assets`](https://github.com/PhyBlock/PhyBlock/tree/main/data/block_assets).

For more detailed information regarding the 3D block asset construction, please refer to Appendix A.1 "Construction of Modular 3D Block Simulation Assets" in our paper.


## 2️⃣ Construction Pipeline of Block Assembly Scenes

![Illustration of Difficulty Levels in Block Assembly Tasks](https://github.com/PhyBlock/PhyBlock/blob/main/Imgs/PhyBlock_Levels.png)

We construct a wide range of 3D block assembly scenes with varying configurations and difficulty levels using the simulated modular block kit. The pipeline consists of four key stages:


### 🔹 1. Collecting Style Reference Images

We curated a diverse set of block assembly images from online sources as references for designing simulation scenes. These images help inspire both structural diversity and layout aesthetics in our block-based environments.


### 🔹 2. Manual Scene Construction and Annotation

Each scene is composed of multiple blocks with varied geometries and colors. To ensure high-quality supervision, we manually construct each scene and annotate the spatial pose (position and orientation) and topological relationships (e.g., stacking dependencies) for all blocks.
All annotations are stored in structured JSON format, with one JSON file per scene. These files can be accessed in the [`data/SCENEs_400_Goal_Jsons`](https://github.com/PhyBlock/PhyBlock/tree/main/data/SCENEs_400_Goal_Jsons).


### 🔹 3. Data Augmentation & Difficulty Classification

We initially created 150 unique block scenes and applied geometric augmentations (e.g., rotation, translation) to expand the dataset to 400 scenes.
Each scene is categorized into one of four difficulty levels based on structural complexity and block dependencies:

* **Level-1**: 36 scenes
* **Level-2**: 121 scenes
* **Level-3**: 138 scenes
* **Level-4**: 119 scenes

*Note: Levels 1 and 2 have partial overlap.*


### 🔹 4. Rendering via Simulation

Using the [Genesis](https://genesis-embodied-ai.github.io/) simulation platform, we render each scene under six different background environments and camera viewpoints. The resulting multi-view images form the basis for our vision-language datasets and question-answering tasks.
