package com.oshimamasara.ai_abc01;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import butterknife.BindView;
import butterknife.ButterKnife;

public class MainActivity extends AppCompatActivity {
    private final String TAG = this.getClass().getSimpleName();

    private static final int PIXEL_WIDTH = 128;
    private AI abcClassifier;


    //private PaintView paintView;

    @BindView(R.id.predict_button)
    View predictButton;

    @BindView(R.id.clear_button)
    View clearButton;

    @BindView(R.id.text_result)
    TextView mResultText;

    @BindView(R.id.canvas)
    PaintView paintView;

    @BindView(R.id.preview_image)
    ImageView previewImage;

    @BindView(R.id.inference_preview)
    LinearLayout inferencePreview;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setTitle(R.string.app_name);
        ButterKnife.bind(this);
        abcClassifier = new AI(this);

        //paintView = (PaintView) findViewById(R.id.canvas);
        DisplayMetrics metrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(metrics);
        paintView.onSizeChanged(metrics);

        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onPredictClicked();
            }
        });

        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onClearClicked();
            }
        });
    }

    private void onPredictClicked() {
        inferencePreview.setVisibility(View.VISIBLE);
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(paintView.getBitmap(), PIXEL_WIDTH, PIXEL_WIDTH, false);
        int digit = abcClassifier.classify(scaledBitmap);

        previewImage.setImageBitmap(scaledBitmap);
        if (digit >= 0) {
            Log.d(TAG, "Found Digit = " + digit);

            String[] answere = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
            String predict = answere[digit] ;

            mResultText.setText(getString(R.string.found_digits, predict));
        } else {
            mResultText.setText(getString(R.string.not_detected));
        }


    }


    public void onClearClicked() {
        paintView.clear();
        mResultText.setText("");
        previewImage.setImageBitmap(null);
    }


}
