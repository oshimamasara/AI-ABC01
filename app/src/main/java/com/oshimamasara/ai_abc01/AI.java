//mnistOutput

package com.oshimamasara.ai_abc01;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.nio.MappedByteBuffer;

public class AI {

    private final String TAG = this.getClass().getSimpleName();

    private Interpreter tflite;
    private ByteBuffer inputBuffer = null;
    private float[][] abcOutput = null;
    private static final String MODEL_PATH = "abc.tflite";
    private static final int NUMBER_LENGTH = 26;
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_IMG_SIZE_X = 128;
    private static final int DIM_IMG_SIZE_Y = 128;
    private static final int DIM_PIXEL_SIZE = 1;
    private static final int BYTE_SIZE_OF_FLOAT = 4;

    public AI(Activity activity){
        try{
            tflite = new Interpreter(loadModelFile(activity));
            inputBuffer =
                    ByteBuffer.allocateDirect(
                            BYTE_SIZE_OF_FLOAT * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);

            inputBuffer.order(ByteOrder.nativeOrder());
            abcOutput = new float[DIM_BATCH_SIZE][NUMBER_LENGTH];
            Log.d(TAG, "Created a Tensorflow Lite Classifier.");
            Log.d(TAG, String.valueOf(abcOutput));
        } catch (IOException e) {
            Log.e(TAG, "IOException loading the tflite file");
        }
    }

    protected void runInference() {
        tflite.run(inputBuffer, abcOutput);
    }

    public int classify(Bitmap bitmap) {
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
        }
        preprocess(bitmap);
        runInference();
        return postprocess();
    }

    private int postprocess() {
        ArrayList<Float> predict = new ArrayList<>();

        for (int i = 0; i < abcOutput[0].length; i++) {
            float value = abcOutput[0][i];
            Log.d(TAG, "Output for " + Integer.toString(i) + ": " + Float.toString(value));
            predict.add(value);
            Log.d(TAG, "predict★ :  " + predict);
        }
        Log.d(TAG, "predict最終★ :  " + predict);

        Float out = predict.get(0);
        int index = 0;

        for (int e = 0; e < predict.size(); e++)
        {
            if (out < predict.get(e))
            {
                out = predict.get(e);
                index = e;
            }
        }
        Log.d(TAG, "最大値★ :  " + out);
        Log.d(TAG, "INDEX★ :  " + index);
        return index;
    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException  {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void preprocess(Bitmap bitmap) {
        if (bitmap == null || inputBuffer == null) {
            return;
        }

        inputBuffer.rewind();

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        long startTime = SystemClock.uptimeMillis();

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < pixels.length; ++i) {
            int pixel = pixels[i];
            int channel = pixel & 0xff;
            inputBuffer.putFloat(0xff - channel);
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Time cost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }
}
