package com.asaunin.neural.network.ocr;

import com.asaunin.neural.network.model.Vector;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageReader {

    private byte[][] pixels;
    private int width;
    private int height;

    public ImageReader read(File file) throws IOException {
        final BufferedImage image = ImageIO.read(file);
        width = image.getWidth();
        final byte[][] pixels = new byte[width][];
        for (int x = 0; x < width; x++) {
            height = image.getHeight();
            pixels[x] = new byte[height];
            for (int y = 0; y < image.getHeight(); y++) {
                pixels[x][y] = (byte) (image.getRGB(x, y) == 0xFFFFFFFF ? 0 : 1);
            }
        }
        this.pixels = pixels;
        return this;
    }

    public Vector toVector() {
        final int[] temp = new int[width * height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                temp[i * width + j] = pixels[i][j];
            }
        }

        return Vector.of(temp);
    }

}
