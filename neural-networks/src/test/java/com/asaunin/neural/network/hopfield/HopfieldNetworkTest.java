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
                .train(Vector.of(1, 0, 1, 0))
                .train(Vector.of(1, 1, 1, 1));
    }

    @Test
    public void whenPatternMatchingThanReturnTrue() {
        assertTrue(network.validate(Vector.of(1, 0, 1, 0)));
        assertTrue(network.validate(Vector.of(1, 1, 1, 1)));
    }

    @Test
    public void whenPatternMismatchingThanReturnFalse() throws Exception {
        assertFalse(network.validate(Vector.of(0, 0, 0, 1)));
        assertFalse(network.validate(Vector.of(0, 0, 1, 0)));
        assertFalse(network.validate(Vector.of(0, 0, 1, 1)));
        assertFalse(network.validate(Vector.of(0, 1, 0, 0)));
        assertFalse(network.validate(Vector.of(0, 1, 1, 0)));
        assertFalse(network.validate(Vector.of(0, 1, 1, 1)));
        assertFalse(network.validate(Vector.of(1, 0, 0, 0)));
        assertFalse(network.validate(Vector.of(1, 0, 0, 1)));
        assertFalse(network.validate(Vector.of(1, 0, 1, 1)));
        assertFalse(network.validate(Vector.of(1, 1, 0, 0)));
        assertFalse(network.validate(Vector.of(1, 1, 0, 1)));
    }

    @Test
    @Ignore("Here comes Hopfield network problem, cause it recognizes inverse patterns")
    public void whenInversePatternMatchingThanReturnFalse() throws Exception {
        assertFalse(network.validate(Vector.of(0, 0, 0, 0)));
        assertFalse(network.validate(Vector.of(0, 1, 0, 1)));
    }

}