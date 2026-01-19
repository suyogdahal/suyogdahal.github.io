"""
Word2Vec Analogy Visualization - 2D Vector Space
Run with: manim -pqh --format png word2vec.py Word2VecAnalogy
"""

from manim import *


class Word2VecAnalogy(Scene):
    def construct(self):
        # Colors
        QUEEN_COLOR = "#e91e63"  # Pink
        WOMAN_COLOR = "#9c27b0"  # Purple
        MAN_COLOR = "#2196f3"    # Blue
        KING_COLOR = "#ff9800"   # Orange
        RESULT_COLOR = "#4caf50" # Green

        # Create coordinate system
        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=7,
            y_length=7,
            axis_config={"color": GRAY, "include_ticks": False},
        )
        axes.move_to(ORIGIN)

        # Define vector positions (simplified 2D representation)
        # queen = woman + royalty_direction
        # king = man + royalty_direction
        # So: queen - woman + man = king
        
        origin = axes.c2p(0, 0)
        queen_end = axes.c2p(3, 4)
        woman_end = axes.c2p(1, 3)
        man_end = axes.c2p(2, 1)
        king_end = axes.c2p(4, 2)
        
        # Vector: queen (from origin)
        queen_vec = Arrow(origin, queen_end, buff=0, color=QUEEN_COLOR, stroke_width=4)
        queen_label = Text("queen", font_size=24, color=QUEEN_COLOR)
        queen_label.next_to(queen_end, UP + RIGHT, buff=0.1)

        # Vector: -woman (shown as dashed from queen tip, going backwards)
        # queen - woman means: start at queen, go in opposite direction of woman
        woman_vec_neg_start = queen_end
        woman_vec_neg_end = axes.c2p(3 - 1, 4 - 3)  # queen - woman
        woman_vec = Arrow(woman_vec_neg_start, woman_vec_neg_end, buff=0, color=WOMAN_COLOR, stroke_width=4)
        woman_label = Text("-woman", font_size=24, color=WOMAN_COLOR)
        woman_label.next_to(
            (np.array(woman_vec_neg_start) + np.array(woman_vec_neg_end)) / 2,
            LEFT, buff=0.2
        )

        # Vector: +man (from the result of queen-woman)
        man_vec_start = woman_vec_neg_end
        man_vec_end = axes.c2p(3 - 1 + 2, 4 - 3 + 1)  # queen - woman + man
        man_vec = Arrow(man_vec_start, man_vec_end, buff=0, color=MAN_COLOR, stroke_width=4)
        man_label = Text("+man", font_size=24, color=MAN_COLOR)
        man_label.next_to(
            (np.array(man_vec_start) + np.array(man_vec_end)) / 2,
            DOWN, buff=0.2
        )

        # The result should be close to king
        # Result vector from origin to final point
        result_end = man_vec_end  # This is queen - woman + man
        result_vec = Arrow(origin, result_end, buff=0, color=RESULT_COLOR, stroke_width=5)
        
        # King vector (for comparison - should be very close to result)
        king_vec = Arrow(origin, king_end, buff=0, color=KING_COLOR, stroke_width=4, max_stroke_width_to_length_ratio=10)
        king_label = Text("king", font_size=24, color=KING_COLOR)
        king_label.next_to(king_end, RIGHT, buff=0.15)

        # Approx label near the result
        approx_label = MathTex(r"\approx", font_size=36, color=WHITE)
        approx_label.next_to(result_end, UP + LEFT, buff=0.15)

        # Formula at bottom
        formula = MathTex(
            r"\vec{queen} - \vec{woman} + \vec{man} \approx \vec{king}",
            font_size=32,
            color=WHITE
        )
        formula.to_edge(DOWN, buff=0.5)

        # Add everything with animations
        self.add(axes)
        
        self.play(GrowArrow(queen_vec), FadeIn(queen_label))
        self.wait(0.5)
        
        self.play(GrowArrow(woman_vec), FadeIn(woman_label))
        self.wait(0.5)
        
        self.play(GrowArrow(man_vec), FadeIn(man_label))
        self.wait(0.5)
        
        self.play(GrowArrow(king_vec), FadeIn(king_label), FadeIn(approx_label))
        self.wait(0.5)
        
        self.play(FadeIn(formula))
        self.wait(1.5)


if __name__ == "__main__":
    print("Run with: manim -pqh --format png word2vec.py Word2VecAnalogy")
