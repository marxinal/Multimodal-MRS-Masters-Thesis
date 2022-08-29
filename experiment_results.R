# Experiment Results

## Participant 1 (compare conditions: audio, comb, random)

# First test (H1): Control(random) vs Experimental (audio-only)
prop.test(x = c(149, 86), n = c(239, 239), alternative = 'greater')

# Second test (H2): Control(random) vs Experimental (multimodal)
prop.test(x = c(148, 86), n = c(239, 239), alternative = 'greater')

# Third t-test: Experimental (only audio) vs Experimental (audio + lyrics)
prop.test(x = c(149, 148), n = c(239, 239), alternative = 'greater')

## Participant 2 (compare conditions: audio, comb, genre)

# First test (H1): Control (genre) vs Experimental (audio-only)
prop.test(x = c(79, 76), n = c(172, 172), alternative = 'greater')

# Second test (H2): Control (genre) vs Experimental (multimodal)
prop.test(x = c(94, 76), n = c(172, 172), alternative = 'greater')

# Third t-test: Experimental (only audio) vs Experimental (audio + lyrics)
prop.test(x = c(94, 79), n = c(172, 172), alternative = 'greater')

