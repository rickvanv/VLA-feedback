# TODOs

- [x] Prepare `compare_feedback_sequential.py` for 2000 training steps (instead of 520) without occlusions
- [x] Run `compare_feedback_sequential.py` without occlusions with 2000 training steps rather than 520
- [x] Do the same but including occlusions
- [x] Implement 4D feedback state-space & 4-branch FFN basis mechanism (`NUM_CHOICES = 4` + gated residual state update)
- [x] Bottleneck the recurrent state to a true 4D vector $s_t \in [-1, 1]^4$ (`138,081` params) so it cannot overfit 64D episode embeddings
- [x] Run `compare_feedback_sequential.py` with the true 4D state bottleneck (without occlusions, 2000 steps)
- [x] Run `compare_feedback_sequential.py` with the true 4D state bottleneck (including occlusions, 2000 steps)





