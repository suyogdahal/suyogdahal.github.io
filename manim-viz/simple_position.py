"""
Simple Positional Encoding - Adding index to embedding
Run with: manim -pql --format gif simple_position.py SimplePositionEncoding
"""

from manim import (
    Scene,
    Text,
    VGroup,
    RoundedRectangle,
    FadeIn,
    Write,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    GRAY,
    WHITE,
)


class SimplePositionEncoding(Scene):
    def construct(self):
        # Colors
        VEC_COLOR = "#64b5f6"       # Blue
        POS_COLOR = "#ff8a65"       # Coral/Orange
        
        words = ["I", "love", "transformers"]
        
        # Row 1: Word embedding vectors
        embed_row = VGroup()
        for word in words:
            vec_box = RoundedRectangle(
                corner_radius=0.1,
                height=1.4,
                width=1.0,
                stroke_color=VEC_COLOR,
                stroke_width=2,
                fill_color=VEC_COLOR,
                fill_opacity=0.15
            )
            # Show some fake embedding values
            vals = VGroup(
                Text("0.3", font_size=16, color=WHITE),
                Text("-0.7", font_size=16, color=WHITE),
                Text("0.2", font_size=16, color=WHITE),
                Text("...", font_size=16, color=GRAY),
            ).arrange(DOWN, buff=0.08)
            vals.move_to(vec_box.get_center())
            
            word_label = Text(f'"{word}"', font_size=18, color=GRAY)
            word_label.next_to(vec_box, UP, buff=0.15)
            
            embed_row.add(VGroup(vec_box, vals, word_label))
        
        embed_row.arrange(RIGHT, buff=0.5)
        embed_label = Text("Word Embeddings", font_size=20, color=VEC_COLOR)
        embed_label.next_to(embed_row, LEFT, buff=0.4)
        
        # Plus sign
        plus = Text("+", font_size=40, color=WHITE)
        
        # Row 2: Position vectors
        pos_row = VGroup()
        for i in range(3):
            vec_box = RoundedRectangle(
                corner_radius=0.1,
                height=1.4,
                width=1.0,
                stroke_color=POS_COLOR,
                stroke_width=2,
                fill_color=POS_COLOR,
                fill_opacity=0.15
            )
            # Position values (all same digit)
            vals = VGroup(
                Text(str(i), font_size=16, color=WHITE),
                Text(str(i), font_size=16, color=WHITE),
                Text(str(i), font_size=16, color=WHITE),
                Text("...", font_size=16, color=GRAY),
            ).arrange(DOWN, buff=0.08)
            vals.move_to(vec_box.get_center())
            
            pos_row.add(VGroup(vec_box, vals))
        
        pos_row.arrange(RIGHT, buff=0.5)
        pos_label = Text("Position Vectors", font_size=20, color=POS_COLOR)
        
        # Equals sign
        equals = Text("=", font_size=40, color=WHITE)
        
        # Row 3: Result vectors (position-encoded embeddings)
        RESULT_COLOR = "#81c784"  # Green
        result_row = VGroup()
        result_vals = [
            ["0.3", "-0.7", "0.2"],   # 0 + embed
            ["1.3", "0.3", "1.2"],    # 1 + embed
            ["2.3", "1.3", "2.2"],    # 2 + embed
        ]
        for i, word in enumerate(words):
            vec_box = RoundedRectangle(
                corner_radius=0.1,
                height=1.4,
                width=1.0,
                stroke_color=RESULT_COLOR,
                stroke_width=2,
                fill_color=RESULT_COLOR,
                fill_opacity=0.15
            )
            vals = VGroup(
                Text(result_vals[i][0], font_size=16, color=WHITE),
                Text(result_vals[i][1], font_size=16, color=WHITE),
                Text(result_vals[i][2], font_size=16, color=WHITE),
                Text("...", font_size=16, color=GRAY),
            ).arrange(DOWN, buff=0.08)
            vals.move_to(vec_box.get_center())
            
            word_label = Text(f'"{word}"', font_size=18, color=GRAY)
            word_label.next_to(vec_box, UP, buff=0.15)
            
            result_row.add(VGroup(vec_box, vals, word_label))
        
        result_row.arrange(RIGHT, buff=0.5)
        result_label = Text("Position-Encoded Word Vectors", font_size=20, color=RESULT_COLOR)
        
        # Create left side (embed + pos) and right side (result)
        left_group = VGroup()
        embed_row.move_to(ORIGIN)
        plus.next_to(embed_row, DOWN, buff=0.2)
        pos_row.next_to(plus, DOWN, buff=0.2)
        left_group.add(embed_row, plus, pos_row)
        left_group.move_to(ORIGIN + LEFT * 2.5)
        
        embed_label.next_to(embed_row, LEFT, buff=0.3)
        pos_label.next_to(pos_row, LEFT, buff=0.3)
        
        equals.next_to(left_group, RIGHT, buff=0.4)
        result_row.next_to(equals, RIGHT, buff=0.4)
        result_label.next_to(result_row, DOWN, buff=0.3)
        
        # Animate
        self.play(FadeIn(embed_row), FadeIn(embed_label), run_time=0.6)
        self.wait(0.3)
        self.play(Write(plus), run_time=0.3)
        self.play(FadeIn(pos_row), FadeIn(pos_label), run_time=0.6)
        self.wait(0.3)
        self.play(Write(equals), run_time=0.3)
        self.play(FadeIn(result_row), FadeIn(result_label), run_time=0.6)
        self.wait(1.5)
