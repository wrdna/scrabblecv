package com.boldgerg;

import static android.Manifest.permission.CAMERA;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;

import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.imgproc.Moments;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

//import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class BoardParserFragment extends Fragment implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "Lexiconv::BoardParser";
    private static final String DEBUGTAG = "LexiconvDB";
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 200;
    private CameraBridgeViewBase mOpenCvCameraView;

    // _ is empty, * is blank
    // _ will not be detected, used to shift when masking for score
    private static final String[] CLASS_NAMES = {
            "_","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O",
            "P","Q","R","S","T","U","V","W","X","Y","Z","*"
    };
    private static final boolean DEBUG_CLASSIFY = true;
    private static final int GRID_SZ = 15;
    private static final int BOARD_PX = 640;
    private static final int TOTAL_CELLS = GRID_SZ * GRID_SZ;
    private static final int CELL_PX = BOARD_PX / GRID_SZ;
    private static final float STEP_PX = BOARD_PX / (float) GRID_SZ;     // 42.666…
    private Mat detectionFrame;
    private Mat orientedBoard;
    private Mat mask;
    private boolean is4Corners = false;
    private Mat[] cells = new Mat[TOTAL_CELLS];

    private static final int[] tilesOnBoard = new int[GRID_SZ * GRID_SZ];
    private static final int[] boardMask = new int[GRID_SZ * GRID_SZ];
    static ArrayList<Integer> filledCells = new ArrayList<>();

    private final TFLiteRuntime CornerDetectTF = new TFLiteRuntime();
    private final TFLiteRuntime TileClassifierTF = new TFLiteRuntime();
    private List<Point> corners = new ArrayList<>();
    private GameStateViewModel viewModel;

    private class TFLiteRuntime {
        private Interpreter interpreter;
        private int inputWidth;
        private int inputHeight;
        private int numInputElements;
        private Size inputSize;
        private Mat blob;
        private Mat flatBlob;
        private ByteBuffer inputBuffer;
        private float[] tempFloatArray;
        private int modelFile;
        private int[] inputShape;
        private int[] outputShape;
        private boolean useGPU = true;

        private void loadModel() {
            try {
                InputStream inputStream = getResources().openRawResource(modelFile);
                byte[] modelBytes = new byte[inputStream.available()];
                inputStream.read(modelBytes);
                inputStream.close();
                Log.d(TAG, "Read TFLite model");

                File tempModelFile = File.createTempFile(
                        "model",
                        ".tflite",
                        BoardParserFragment.this.requireActivity().getCacheDir());

                FileOutputStream fos = new FileOutputStream(tempModelFile);
                fos.write(modelBytes);
                fos.close();

                Interpreter.Options options = new Interpreter.Options();

                if (useGPU) {
                    options.addDelegate(new GpuDelegate());
                    Log.i(TAG, "TFLite GPU Delegate enabled");
                } else {
                    //options.setNumThreads(4);
                    //options.setUseNNAPI(true);
                }

                interpreter = new Interpreter(tempModelFile, options);
                Log.i(TAG, "TFLite model loaded successfully");

                int numInputs = interpreter.getInputTensorCount();
                int numOutputs = interpreter.getOutputTensorCount();

                if (numInputs > 0) {
                    Tensor inputTensor = interpreter.getInputTensor(0);
                    inputShape = inputTensor.shape();
                    Log.d(TAG, "TFLite input shape: " + Arrays.toString(inputShape));
                }

                if (numOutputs > 0) {
                    Tensor outputTensor = interpreter.getOutputTensor(0);
                    outputShape = outputTensor.shape();
                    Log.d(TAG, "TFLite output shape: " + Arrays.toString(outputShape));
                }
            } catch (Exception e) {
                Log.e(TAG, "Failed to load TFLite model: " + e.getMessage());
                e.printStackTrace();
            }
        }

        private void initTFLiteRuntime() {
            try {
                // Load the model
                loadModel();

                inputBuffer = ByteBuffer.allocateDirect(numInputElements * 4); // 4 bytes per float
                inputBuffer.order(ByteOrder.nativeOrder());

                Log.d(TAG, "TFLite Runtime initialized successfully");
            } catch (Exception e) {
                Log.e(TAG, "Failed to initialize TFLite Runtime: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    private static class Detection {
        float x1, y1, x2, y2, confidence;
        int classId;

        Detection(float[] arr, boolean normalized, int imgW, int imgH) {
            if (normalized) {
                this.x1 = arr[0] * imgW;
                this.y1 = arr[1] * imgH;
                this.x2 = arr[2] * imgW;
                this.y2 = arr[3] * imgH;
            } else {
                this.x1 = arr[0];
                this.y1 = arr[1];
                this.x2 = arr[2];
                this.y2 = arr[3];
            }
            this.confidence = arr[4];
            this.classId = (int) arr[5];
        }
    }



    private static List<Point> getTrueCorners(List<Detection> detections) {
        List<Detection> filtered = new ArrayList<>();
        int PX_FILTER_DIST_SQ = 300;
        float CONFIDENCE_THRESH = 0.03f;

        for (Detection det : detections) {
            int centerX = Math.round(det.x1 + (det.x2 - det.x1) / 2.0f);
            int centerY = Math.round(det.y1 + (det.y2 - det.y1) / 2.0f);
            boolean tooClose = false;

            for (Detection kept : filtered) {
                int keptX = Math.round(kept.x1 + (kept.x2 - kept.x1) / 2.0f);
                int keptY = Math.round(kept.y1 + (kept.y2 - kept.y1) / 2.0f);
                int dx = centerX - keptX;
                int dy = centerY - keptY;
                //Log.d(TAG, String.format("%d, %d", keptX, keptY));

                if ((dx * dx + dy * dy) <= PX_FILTER_DIST_SQ) {
                    tooClose = true;
                    break;
                }
            }

            if (det.confidence >= CONFIDENCE_THRESH && !tooClose) {
                filtered.add(det);
            }
        }

        int limit = Math.min(4, filtered.size()); // keep 4 corners w/highest confidence
        List<Point> corners = new ArrayList<>(limit);

        for (int i = 0; i < limit; i++) {
            Detection det = filtered.get(i);
            int centerX = Math.round(det.x1 + (det.x2 - det.x1) / 2.0f);
            int centerY = Math.round(det.y1 + (det.y2 - det.y1) / 2.0f);
            corners.add(new Point(centerX, centerY));
        }

        return corners;
    }

    private void setupCornerDetection() {
        int inputWidth = 640;
        int inputHeight = 640;
        Size inputSize = new Size(inputWidth, inputHeight);
        int numInputElements = 1 * 3 * inputHeight * inputWidth;

        // TFLite Setup
        CornerDetectTF.inputWidth = inputWidth;
        CornerDetectTF.inputHeight = inputHeight;
        CornerDetectTF.inputSize = inputSize;
        CornerDetectTF.numInputElements = numInputElements;
        CornerDetectTF.blob = new Mat();
        CornerDetectTF.flatBlob = new Mat();
        CornerDetectTF.tempFloatArray = new float[numInputElements];
        CornerDetectTF.modelFile = R.raw.webcam_corners_yv10_fp16;

        CornerDetectTF.useGPU = true;
        CornerDetectTF.initTFLiteRuntime();
    }

    private void setupTileClassification() {
        int inputWidth = 42;
        int inputHeight = 42;
        Size inputSize = new Size(inputWidth, inputHeight);
        int numInputElements = 1 * 3 * inputHeight * inputWidth;

        // TFLite Setup
        TileClassifierTF.inputWidth = inputWidth;
        TileClassifierTF.inputHeight = inputHeight;
        TileClassifierTF.inputSize = inputSize;
        TileClassifierTF.numInputElements = numInputElements;
        TileClassifierTF.blob = new Mat();
        TileClassifierTF.flatBlob = new Mat();
        TileClassifierTF.tempFloatArray = new float[numInputElements];
        TileClassifierTF.modelFile = R.raw.webcam_tile_e199;

        CornerDetectTF.useGPU = false;
        TileClassifierTF.initTFLiteRuntime();

        detectionFrame = new Mat();
        orientedBoard = new Mat();
        mask = new Mat();
    }

    @Override
    public View onCreateView(LayoutInflater inf, ViewGroup c, Bundle s) {
        View view = inf.inflate(R.layout.fragment_boardparser, c, false);
        viewModel = new ViewModelProvider(requireActivity()).get(GameStateViewModel.class);

        final boolean[] hasNavigated = {false};

        // Not a fan of this being here, should probably have a lot of this logic in GameStateViewModel
        Button scoreShutter = view.findViewById(R.id.shutterButton);
        scoreShutter.setEnabled(false);
        scoreShutter.setOnClickListener(v -> {
                if (!hasNavigated[0]) {
                    if (viewModel.isNewBoardStateValid()) {
                        viewModel.scoreMove();
                        hasNavigated[0] = true;
                        NavController nav = NavHostFragment.findNavController(BoardParserFragment.this);
                        nav.navigate(R.id.action_boardParserFragment_to_scoreBoardFragment);
                    }
                }
        });
        viewModel.getIsBoardOriented().observe(getViewLifecycleOwner(), enabled -> {
            scoreShutter.setEnabled(enabled);
        });

        mOpenCvCameraView = view.findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMaxFrameSize(1920,1080);

        return view;
    }

    @Override
    public void onViewCreated(View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "OpenCV initialization failed!");
            // (Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG)).show();
            // return null;
        }

        ActivityResultLauncher<String> camPermLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(),
                granted -> { if (granted) startCamera(); }
        );

        if (ContextCompat.checkSelfPermission(requireContext(), CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            Log.d(TAG, "GETTING PERMISSIONS");
            camPermLauncher.launch(CAMERA);
        }
    }

    private void startCamera() {
        mOpenCvCameraView.setCameraPermissionGranted();
        mOpenCvCameraView.enableView();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.enableView();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

        if (TileClassifierTF.interpreter != null) {
            TileClassifierTF.interpreter.close();
        }
        if (CornerDetectTF.interpreter != null) {
            CornerDetectTF.interpreter.close();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        setupCornerDetection();
        setupTileClassification();
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        Mat input = inputFrame.rgba();
        Mat displayFrame = input.clone();

        getCurrentTiles(input, displayFrame);

        return displayFrame;
    }

    public void getCurrentTiles(Mat originalFrame, Mat displayFrame) {
        List<Point> scaledCorners = new ArrayList<>();
        Size originalSize = new Size(originalFrame.width(), originalFrame.height());

        double scaleX, scaleY;
        scaleX = (double) originalSize.width / 640.0;
        scaleY = (double) originalSize.height / 640.0;


        // Using this copy for detection/display
        // Corner detection needs to be on 640x640, we scale those points back to a higher quality
        // image, retaining the resolution before scaling back to 640x640 for classification.

        Imgproc.resize(originalFrame, detectionFrame, new Size(640, 640));

        detectCorners(detectionFrame);

        if (checkCorners(corners)) {
            for (Point corner : corners) {
                Point scaledCorner = new Point(corner.x * scaleX, corner.y * scaleY);
                scaledCorners.add(scaledCorner);
            }
            orientBoardToSize(originalFrame, orientedBoard, scaledCorners, new Size(640, 640));
            classifyTiles(orientedBoard);
            viewModel.setIsBoardOriented(true);
            if (DEBUG_CLASSIFY)
            {
                drawTilesOnBoard(orientedBoard);
                Imgproc.resize(
                        orientedBoard,
                        displayFrame,
                        new Size(displayFrame.width(),
                                displayFrame.height())
                );
            }
        } else {
            viewModel.setIsBoardOriented(false);
            for (Point corner : corners) {
                Point scaledCorner = new Point(corner.x * scaleX, corner.y * scaleY);
                scaledCorners.add(scaledCorner);
                Imgproc.circle(displayFrame, scaledCorner, 5, new Scalar(0, 255, 0), 2);
            }
        }
    }

    public void classifyTiles(Mat orientedBoard) {
        filledCells.clear();
        Arrays.fill(boardMask, 0);
        Arrays.fill(tilesOnBoard, 0);

        int cell_idx = 0;

        for (int i = 0; i < GRID_SZ; i++) {
            int y1 = Math.round(i * STEP_PX);
            int y2 = Math.round((i+1) * STEP_PX);

            for (int j = 0; j < GRID_SZ; j++) {
                int x1 = Math.round(j* STEP_PX);
                int x2 = Math.round((j+1) * STEP_PX);

                x2 = Math.min(x2, BOARD_PX);
                y2 = Math.min(y2, BOARD_PX);

                cell_idx = i * GRID_SZ + j;

                cells[cell_idx] = orientedBoard.submat(new Range(y1, y2), new Range(x1, x2));

                Imgproc.cvtColor(cells[cell_idx], cells[cell_idx], Imgproc.COLOR_RGBA2RGB);

                Scalar lower = new Scalar(0, 0, 0);
                Scalar upper = new Scalar(110, 110, 110);

                Core.inRange(cells[cell_idx], lower, upper, mask);

                double coverage = Core.countNonZero(mask) / (double) (mask.rows() * mask.cols());

                if (coverage >= 0.3) {
                    filledCells.add(cell_idx);
                }
            }
        }

        // TODO: Filled cells will become newCells, we need to only classify newly placed tiles
        // TODO: ^^^^ LOOKING AT THIS... IDK IF I CAN DO THIS NOW, NEED TO CLASSIFY ALL TILES OT GET BOARD ORIENTATION
        for (int i = 0; i < filledCells.size(); i++) {
            Mat rgb = new Mat();
            Imgproc.resize(cells[filledCells.get(i)], rgb, new Size(42, 42));
            rgb.convertTo(rgb, CvType.CV_32FC3);

            rgb.get(0, 0, TileClassifierTF.tempFloatArray);

            // feed the interpreter

            TileClassifierTF.inputBuffer.rewind();
            TileClassifierTF.inputBuffer.asFloatBuffer().put(TileClassifierTF.tempFloatArray);

            float[][] output = new float[1][27];
            TileClassifierTF.interpreter.run(TileClassifierTF.inputBuffer, output);

            int predictedIdx = argmax(output[0]);
            float confidence = max(output[0]);

            if (confidence >= 0.4f) {
                // network predicts 0-26, but board represents an empty cell as 0
                // must add 1 to shift
                tilesOnBoard[filledCells.get(i)] = predictedIdx + 1;
                boardMask[filledCells.get(i)] = 1;
            }
        }
        viewModel.setNewBoardMask(boardMask);
        viewModel.setNewBoardState(tilesOnBoard);
    }

    public static void drawTilesOnBoard(Mat orientedBoard) {
        // for (int i = 0; i < GRID_SZ; i++) {
        //     int y1 = Math.round(i * STEP_PX);
        //     int y2 = Math.round((i + 1) * STEP_PX);

        //     for (int j = 0; j < GRID_SZ; j++) {
        //         int x1 = Math.round(j * STEP_PX);
        //         int x2 = Math.round((j + 1) * STEP_PX);

        //         Imgproc.rectangle(orientedBoard, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 0, 255), 1);
        //     }
        // }

        for (int i = 0; i < filledCells.size(); i++) {
            if (filledCells.get(i) != 0) {
                drawTile(orientedBoard, filledCells.get(i));
            }
        }
    }

    public static void drawTile(Mat img, int tileIdx) {

        int gridSize = 15;
        int tilePx = 640 / gridSize;

        int row = tileIdx / gridSize;
        int col = tileIdx % gridSize;

        int x = col * tilePx + 10;
        int y = row * tilePx + 30;

        Scalar color = new Scalar(0, 255, 0);
        int thickness = 2;
        double fontSize = 0.6;

        if (tileIdx >= 0 && tileIdx < tilesOnBoard.length) {
            int classIndex = tilesOnBoard[tileIdx];
            if (classIndex >= 0 && classIndex < CLASS_NAMES.length) {
                Imgproc.putText(
                        img,
                        CLASS_NAMES[classIndex],
                        new Point(x, y),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        fontSize,
                        color,
                        thickness
                );
            }
        }
    }

    public void detectCorners(Mat inputFrame) {
        Mat rgbFrame = new Mat();
        float[][][] output = new float[0][][];
        List<Detection> topBoxes = new ArrayList<>();

        Imgproc.cvtColor(inputFrame, rgbFrame, Imgproc.COLOR_RGBA2RGB);

        // TFLite corner detection
        Mat work = new Mat();
        Imgproc.cvtColor(rgbFrame, work, Imgproc.COLOR_RGBA2RGB);
        work.convertTo(work, CvType.CV_32FC3, 1.0 / 255.0);

        int elements = (int) (work.total() * work.channels());
        if (CornerDetectTF.tempFloatArray == null ||
                CornerDetectTF.tempFloatArray.length != elements)
        {
            CornerDetectTF.tempFloatArray = new float[elements];
        }
        work.get(0, 0, CornerDetectTF.tempFloatArray);

        CornerDetectTF.inputBuffer.rewind();
        CornerDetectTF.inputBuffer.asFloatBuffer()
                .put(CornerDetectTF.tempFloatArray);

        float[][][] raw = new float[1][300][6];
        CornerDetectTF.interpreter.run(CornerDetectTF.inputBuffer, raw);

        for (int i = 0; i < Math.min(raw[0].length, 32); ++i) {
            topBoxes.add(new Detection(raw[0][i], false,
                    CornerDetectTF.inputWidth,
                    CornerDetectTF.inputHeight));
        }
        // Will be in space of 640x640
        corners = getTrueCorners(topBoxes);
    }

    private boolean checkCorners(List<Point> corners) {
        // Must have 4 corners
        if (corners.size() != 4) return false;

        final double MIN_PX_DIST_SQ = 15000;
        double[] sums = new double[4];
        double[] diffs = new double[4];
        for (int i = 0; i < 4; i++) {
            Point pt = corners.get(i);
            sums[i] = pt.x + pt.y;
            diffs[i] = pt.x - pt.y;
        }

        int topLeftIdx = indexOfMin(sums);
        int topRightIdx = indexOfMin(diffs);
        int bottomRightIdx = indexOfMax(sums);
        int bottomLeftIdx = indexOfMax(diffs);

        Point tl = corners.get(topLeftIdx);
        Point tr = corners.get(topRightIdx);
        Point br = corners.get(bottomRightIdx);
        Point bl = corners.get(bottomLeftIdx);
        int theta_tr = (int) calculateAngle(tl, tr, bl);
        int theta_br = (int) calculateAngle(tr, br, bl);
        int theta_bl = (int) calculateAngle(br, bl, tl);
        int theta_tl = (int) calculateAngle(bl, tl, tr);

        // Convex quadrilateral check
        if (!(theta_tr <= 130 &&
                theta_br <= 130 &&
                theta_bl <= 130 &&
                theta_tl <= 130))
        {
            return false;
        }

        // Min pixel distance check
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                Point a = corners.get(i);
                Point b = corners.get(j);
                double dx = a.x - b.x;
                double dy = a.y - b.y;
                if ((dx * dx + dy * dy) < MIN_PX_DIST_SQ) return false;
            }
        }

        return true;
    }

    private void orientBoardToSize(Mat inputFrame, Mat orientedBoard, List<Point> sourceCorners, Size targetSize) {
        double[] sums = new double[4];
        double[] diffs = new double[4];
        for (int i = 0; i < 4; i++) {
            Point pt = sourceCorners.get(i);
            sums[i] = pt.x + pt.y;
            diffs[i] = pt.x - pt.y;
        }

        int topLeftIdx = indexOfMin(sums);
        int topRightIdx = indexOfMin(diffs);
        int bottomRightIdx = indexOfMax(sums);
        int bottomLeftIdx = indexOfMax(diffs);

        // Source points in the original image
        MatOfPoint2f srcPts = new MatOfPoint2f(
                sourceCorners.get(topLeftIdx),
                sourceCorners.get(bottomLeftIdx),
                sourceCorners.get(bottomRightIdx),
                sourceCorners.get(topRightIdx)
        );

        // Destination points in target (640x640)
        MatOfPoint2f dstPts = new MatOfPoint2f(
                new Point(0, 0),
                new Point(targetSize.width - 1, 0),
                new Point(targetSize.width - 1, targetSize.height - 1),
                new Point(0, targetSize.height - 1)
        );

        Mat transform = Imgproc.getPerspectiveTransform(srcPts, dstPts);

        Imgproc.warpPerspective(inputFrame, orientedBoard, transform, targetSize);
    }

    public static double calculateAngle(Point a, Point b, Point c) {
        double angle1 = Math.atan2(a.y - b.y, a.x - b.x);
        double angle2 = Math.atan2(c.y - b.y, c.x - b.x);

        double angle = Math.abs(angle1 - angle2);
        if (angle > Math.PI) angle = 2 * Math.PI - angle;

        return Math.toDegrees(angle);
    }

    public static int argmax(float[] arr) {
        int maxIdx = 0;
        float maxVal = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    public static float max(float[] arr) {
        float maxVal = arr[0];
        for (float v : arr) {
            if (v > maxVal) maxVal = v;
        }
        return maxVal;
    }

    private int indexOfMin(double[] arr) {
        int minIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] < arr[minIdx]) {
                minIdx = i;
            }
        }
        return minIdx;
    }

    private int indexOfMax(double[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}