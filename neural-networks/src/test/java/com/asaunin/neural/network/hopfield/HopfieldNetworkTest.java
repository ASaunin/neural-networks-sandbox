package com.asaunin.neural.network.hopfield;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@DisplayName("Hopfield network example")
class HopfieldNetworkTest {

    private HopfieldNetwork network;

    @BeforeEach
    void init() {
        network = new HopfieldNetwork(4)
                .train(NormalizedVector.of(1, 0, 1, 0))
                .train(NormalizedVector.of(1, 1, 1, 1));
    }

    @DisplayName("When pattern is matching, than return true")
    @Test
    void whenPatternMatchingThanReturnTrue() {
        assertTrue(network.predict(NormalizedVector.of(1, 0, 1, 0)));
        assertTrue(network.predict(NormalizedVector.of(1, 1, 1, 1)));
    }

    @DisplayName("When pattern is not matching, than return false")
    @Test
    void whenPatternMismatchingThanReturnFalse() {
        assertFalse(network.predict(NormalizedVector.of(0, 0, 0, 1)));
        assertFalse(network.predict(NormalizedVector.of(0, 0, 1, 0)));
        assertFalse(network.predict(NormalizedVector.of(0, 0, 1, 1)));
        assertFalse(network.predict(NormalizedVector.of(0, 1, 0, 0)));
        assertFalse(network.predict(NormalizedVector.of(0, 1, 1, 0)));
        assertFalse(network.predict(NormalizedVector.of(0, 1, 1, 1)));
        assertFalse(network.predict(NormalizedVector.of(1, 0, 0, 0)));
        assertFalse(network.predict(NormalizedVector.of(1, 0, 0, 1)));
        assertFalse(network.predict(NormalizedVector.of(1, 0, 1, 1)));
        assertFalse(network.predict(NormalizedVector.of(1, 1, 0, 0)));
        assertFalse(network.predict(NormalizedVector.of(1, 1, 0, 1)));
    }

    @DisplayName("When inverted pattern is matching, than return true")
    @Test
    @Disabled("Here comes Hopfield network problem, cause it recognizes inverse patterns")
    void whenInversePatternMatchingThanReturnFalse() {
        assertFalse(network.predict(NormalizedVector.of(0, 0, 0, 0)));
        assertFalse(network.predict(NormalizedVector.of(0, 1, 0, 1)));
    }

}