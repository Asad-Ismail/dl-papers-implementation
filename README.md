# Deep Learning Papers Implementation

Clean implementations of influential deep learning papers, bootstrapped with [DeepCode](https://github.com/HKUDS/DeepCode) and validated through human review.

## Papers

| Paper | Implementation | Paper Link | Status |
|-------|----------------|------------|--------|
| **Farsight: Boosting Vision-Language Models with Long-Term Memory** | [📁 papers/farsight](./papers/farsight) | [Paper](https://arxiv.org/abs/2412.12425) | ✅ |
| **REGLA: Representation-Enhanced Gated Linear Attention** | [📁 papers/regla](./papers/regla) | [Paper](https://arxiv.org/abs/2407.03741) | ✅ |
| **SwiftEdit: Lightning Fast Text-Guided Image Editing** | [📁 papers/swiftedit](./papers/swiftedit) | [Paper (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.html) | 🔄 |

**Legend:** ✅ Reviewed  |  🔄 In Review  |  📋 Planned

## Implementation Notes

**Code Quality & Base Model Dependency:**
Implementation accuracy depends heavily on the base model used for code generation. For instance, using GPT-4o-mini resulted in hallucinated components (e.g., additional positional encodings in Farsight that were not in the paper). Generated code often includes excessive validation logic (dimension checks, tensor conversions, error handling) that obscures the core algorithm. Human review focuses on extracting the essential paper logic from this scaffolding.

**Current Limitations:**

- Generated boilerplate prioritizes safety over readability; core algorithmic logic can be buried under defensive programming patterns
- No user control over implementation preferences (package manager, test coverage, code style, etc.)

**Future Improvements:**

Key improvements that would greatly help DeepCode:

- Standardize project structure and dependency management
- Generate more concise, paper-focused code with minimal boilerplate
- Better prompt engineering to reduce hallucinations and unnecessary abstractions
- Allow user configuration for tooling preferences and code generation style

## License

MIT License - See [LICENSE](LICENSE) file for details