package com.asaunin.neural.network.hopfield;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.assertTrue;

public class HopfieldNetworkTest {

    private HopfieldNetwork network;

    @Before
    public void init() {
        network = new HopfieldNetwork(4)
                .train(NormalizedVector.of(1, 0, 1, 0))
                .train(NormalizedVector.of(1, 1, 1, 1));
    }

    @Test
    public void whenPatternMatchingThanReturnTrue() {
        assertTrue(network.predict(NormalizedVector.of(1, 0, 1, 0)));
        assertTrue(network.predict(NormalizedVector.of(1, 1, 1, 1)));
    }

    @Test
    public void whenPatternMismatchingThanReturnFalse() {
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

    @Test
    @Ignore("Here comes Hopfield network problem, cause it recognizes inverse patterns")
    public void whenInversePatternMatchingThanReturnFalse() {
        assertFalse(network.predict(NormalizedVector.of(0, 0, 0, 0)));
        assertFalse(network.predict(NormalizedVector.of(0, 1, 0, 1)));
    }

}