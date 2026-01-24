"""
Input Embedding Visualization - Token IDs to Embedding Vectors
Run with: manim -pql --format gif embedding.py EmbeddingFlow
"""

from manim import (
    Scene,
    Text,
    VGroup,
    RoundedRectangle,
    Arrow,
    FadeIn,
    GrowArrow,
    Write,
    TransformFromCopy,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    BOLD,
    GRAY,
    WHITE,
)


class EmbeddingFlow(Scene):
    def construct(self):
        # Colors
        TOKEN_COLOR = "#81c784"     # Green
        EMBED_COLOR = "#ffb74d"     # Orange
        ARROW_COLOR = "#90a4ae"     # Gray
        VEC_COLOR = "#64b5f6"       # Blue
        
        # ============ STEP 1: Token IDs (continuing from previous) ============
        token_ids = [42, 891, 2048]
        
        token_boxes = VGroup()
        for tid in token_ids:
            text = Text(str(tid), font_size=28, color=TOKEN_COLOR)
            box = RoundedRectangle(
                corner_radius=0.12,
                height=0.7,
                width=0.9,
                stroke_color=TOKEN_COLOR,
                stroke_width=2,
                fill_color=TOKEN_COLOR,
                fill_opacity=0.15
            )
            text.move_to(box.get_center())
            token_boxes.add(VGroup(box, text))
        
        token_boxes.arrange(RIGHT, buff=0.15)
        
        left_bracket = Text("[", font_size=40, color=TOKEN_COLOR)
        right_bracket = Text("]", font_size=40, color=TOKEN_COLOR)
        left_bracket.next_to(token_boxes, LEFT, buff=0.1)
        right_bracket.next_to(token_boxes, RIGHT, buff=0.1)
        
        token_array = VGroup(left_bracket, token_boxes, right_bracket)
        token_array.to_edge(UP, buff=0.6)
        
        token_label = Text("Token IDs", font_size=20, color=GRAY)
        token_label.next_to(token_array, LEFT, buff=0.3)
        
        self.add(token_array, token_label)
        self.wait(0.3)
        
        # ============ STEP 2: Embedding Lookup Table ============
        embed_layer = RoundedRectangle(
            corner_radius=0.15,
            height=1.0,
            width=4.5,
            stroke_color=EMBED_COLOR,
            stroke_width=2,
            fill_color=EMBED_COLOR,
            fill_opacity=0.2
        )
        embed_text = Text("Embedding Lookup Table", font_size=22, color=EMBED_COLOR)
        embed_text.move_to(embed_layer.get_center())
        embed_group = VGroup(embed_layer, embed_text)
        embed_group.next_to(token_array, DOWN, buff=0.8)
        
        # Arrow from tokens to embedding
        arrow_to_embed = Arrow(
            token_array.get_bottom(),
            embed_group.get_top(),
            buff=0.1,
            color=ARROW_COLOR,
            stroke_width=2
        )
        
        self.play(
            GrowArrow(arrow_to_embed),
            FadeIn(embed_group),
            run_time=0.6
        )
        self.wait(0.5)
        
        # ============ STEP 3: Embedding Vectors Output ============
        # Create embedding vector representations
        embed_vectors = VGroup()
        words = ["I", "love", "transformers"]
        
        for i, word in enumerate(words):
            # Vector box (tall rectangle representing d-dimensional vector)
            vec_box = RoundedRectangle(
                corner_radius=0.1,
                height=1.8,
                width=0.8,
                stroke_color=VEC_COLOR,
                stroke_width=2,
                fill_color=VEC_COLOR,
                fill_opacity=0.2
            )
            
            # Dots to represent vector components
            dots = VGroup()
            values = ["0.2", "...", "-0.5"]
            for val in values:
                dot_text = Text(val, font_size=14, color=WHITE)
                dots.add(dot_text)
            dots.arrange(DOWN, buff=0.2)
            dots.move_to(vec_box.get_center())
            
            # Word label below
            word_label = Text(f'"{word}"', font_size=16, color=GRAY)
            word_label.next_to(vec_box, DOWN, buff=0.15)
            
            embed_vectors.add(VGroup(vec_box, dots, word_label))
        
        embed_vectors.arrange(RIGHT, buff=0.5)
        embed_vectors.next_to(embed_group, DOWN, buff=0.7)
        
        # Arrow from embedding layer to vectors
        arrow_to_vecs = Arrow(
            embed_group.get_bottom(),
            embed_vectors.get_top(),
            buff=0.1,
            color=ARROW_COLOR,
            stroke_width=2
        )
        
        dim_label = Text("d-dimensional vectors", font_size=18, color=GRAY)
        dim_label.next_to(embed_vectors, DOWN, buff=0.3)
        
        self.play(
            GrowArrow(arrow_to_vecs),
            run_time=0.4
        )
        self.play(
            *[FadeIn(ev, shift=UP * 0.2) for ev in embed_vectors],
            run_time=0.8
        )
        self.play(Write(dim_label), run_time=0.4)
        self.wait(1.5)
