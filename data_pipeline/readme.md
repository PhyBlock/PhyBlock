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
