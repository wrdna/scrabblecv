package com.boldgerg;

import android.util.Log;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class GameStateViewModel extends ViewModel {

    private static final String TAG = "Lexiconv::GameStateViewModel";

    private final MutableLiveData<Integer> playerCount = new MutableLiveData<>();
    private final MutableLiveData<List<String>> playerNames = new MutableLiveData<>();
    private final MutableLiveData<Integer> currentTurn = new MutableLiveData<>();
    private final MutableLiveData<int[]> boardState = new MutableLiveData<int[]>();
    private final MutableLiveData<int[]> newBoardState = new MutableLiveData<>();
    private final MutableLiveData<int[]> boardMask = new MutableLiveData<>();
    private final MutableLiveData<int[]> newBoardMask = new MutableLiveData<>();
    private final MutableLiveData<int[]> playerScores = new MutableLiveData<>();
    private final MutableLiveData<Boolean> isBoardOriented = new MutableLiveData<>();
    private final MutableLiveData<Integer> turnCount = new MutableLiveData<>();

    private static final String[] CLASS_NAMES = {
            "_","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O",
            "P","Q","R","S","T","U","V","W","X","Y","Z","*"
    };
    private static final int[] LETTER_POINTS = {
          //_,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q ,R,S,T,U,V,W,X,Y,Z, *
            0,1,3,3,2,1,4,2,4,1,8,5,1,3,1,1,3,10,1,1,1,1,4,4,8,4,10,0
    };

    static final int[][] Lmult = {
            {1,1,1,2,1,1,1,1,1,1,1,2,1,1,1},
            {1,1,1,1,1,3,1,1,1,3,1,1,1,1,1},
            {1,1,1,1,1,1,2,1,2,1,1,1,1,1,1},
            {2,1,1,1,1,1,1,2,1,1,1,1,1,1,2},
            {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
            {1,3,1,1,1,3,1,1,1,3,1,1,1,3,1},
            {1,1,2,1,1,1,2,1,2,1,1,1,2,1,1},
 /*middle*/ {1,1,1,2,1,1,1,1,1,1,1,2,1,1,1},
            {1,1,2,1,1,1,2,1,2,1,1,1,2,1,1},
            {1,3,1,1,1,3,1,1,1,3,1,1,1,3,1},
            {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
            {2,1,1,1,1,1,1,2,1,1,1,1,1,1,2},
            {1,1,1,1,1,1,2,1,2,1,1,1,1,1,1},
            {1,1,1,1,1,3,1,1,1,3,1,1,1,1,1},
            {1,1,1,2,1,1,1,1,1,1,1,2,1,1,1}
    };

    static final int[][] Wmult = {
            {3,1,1,1,1,1,1,3,1,1,1,1,1,1,3},
            {1,2,1,1,1,1,1,1,1,1,1,1,1,2,1},
            {1,1,2,1,1,1,1,1,1,1,1,1,2,1,1},
            {1,1,1,2,1,1,1,1,1,1,1,2,1,1,1},
            {1,1,1,1,2,1,1,1,1,1,2,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
 /*middle*/ {3,1,1,1,1,1,1,2,1,1,1,1,1,1,3},
            {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,2,1,1,1,1,1,2,1,1,1,1},
            {1,1,1,2,1,1,1,1,1,1,1,2,1,1,1},
            {1,1,2,1,1,1,1,1,1,1,1,1,2,1,1},
            {1,2,1,1,1,1,1,1,1,1,1,1,1,2,1},
            {3,1,1,1,1,1,1,3,1,1,1,1,1,1,3}
    };


    private int turnCnt;
    private int[] currBoard;
    private int[] currMask;

    private int[] newBoard;
    private int[] newMask;

    private int[] rotNewBoard;
    private int[] rotNewMask;
    private int[] result;
    private int[] newTilesMask;
    private int[] newTiles;

    public void initNewGame() {
        int[] initBoard = new int[15 * 15];
        Arrays.fill(initBoard, 0);
        setBoardState(initBoard);
        setNewBoardState(initBoard);
        setBoardMask(initBoard);
        setNewBoardMask(initBoard);
        setCurrentTurn(0);
        resetTurnCount();
        initPlayerScores();
    }

    public void setPlayerNames(List<String> names) {
        playerNames.setValue(names);
    }

    public List<String> getPlayerNames() {
        return playerNames.getValue();
    }

    public Integer getCurrentTurn() {
        return currentTurn.getValue();
    }

    public void setCurrentTurn(int turn) {
        currentTurn.setValue(turn);
    }

    public void nextPlayersTurn() {
        setCurrentTurn(getTurnCount() % getPlayerCount());
    }

    public void scoreMove() {
        final int SZ = 15;
        List<int[]> words = extractWordIndices();
        Log.d(TAG, words.toString());
        int total = 0;

        for (int w = 0; w < words.size(); ++w) {
            int[] wordCells = words.get(w);
            int wordSum = 0;
            int wordMul = 1;

            for (int k = 0; k < wordCells.length; ++k) {

                int idx = wordCells[k];
                int r = idx / SZ;
                int c = idx % SZ;

                int letter = rotNewBoard[idx];
                int pts = LETTER_POINTS[letter];

                // multipliers only apply to new tiles
                if (newTilesMask[idx] == 1) {
                    pts *= Lmult[r][c]; // DL / TL
                    wordMul *= Wmult[r][c]; // DW / TW
                }
                wordSum += pts;
            }
            //Log.d(TAG, String.valueOf(wordSum));
            total += wordSum * wordMul;
        }

        // bingo
        int tilesUsed = 0;
        for (int i = 0; i < newTilesMask.length; ++i)
            tilesUsed += newTilesMask[i];

        if (tilesUsed == 7) total += 50;

        int player = getCurrentTurn();
        setPlayerScore(player, getPlayerScore(player) + total);

        iterateTurnCount();
        nextPlayersTurn();
        setBoardState(safeCopy(getNewBoardState(), 225));
        setBoardMask(safeCopy(getNewBoardMask(), 225));
    }

    public boolean isNewBoardStateValid() {
        // if (!Arrays.equals(newBoardState.getValue(),
        //                    boardState.getValue()))
        // {
        // }
        turnCnt = getTurnCount();
        currBoard = safeCopy(getBoardState(), 225);
        currMask = safeCopy(getBoardMask(), 225);

        newBoard = safeCopy(getNewBoardState(), 225);
        newMask = safeCopy(getNewBoardMask(), 225);

        rotNewBoard = Arrays.copyOf(newBoard, 225);
        rotNewMask = Arrays.copyOf(newMask, 225);

        result = new int[currMask.length];
        newTilesMask = new int[currMask.length];
        newTiles = new int[currMask.length];

        if (turnCnt == 0) {
            if (newBoard[7 * 15 + 7] == 0) {
                return false;
            } else {
                newTiles = newBoard.clone();
                newTilesMask = newMask.clone();
            }
        } else {
            int sz = 15;
            // finding matching orientation
            for (int rot = 0; rot < 4; rot++) {
                newTilesMask = new int[currMask.length];
                newTiles = new int[currMask.length];

                if (rot != 0) {
                    for (int i = 0; i < sz; i++) {
                        for (int j = 0; j < sz; j++) {
                            int idx = (i * sz) + j;
                            int idx_rot = (j * sz) + (sz - i - 1);
                            rotNewBoard[idx_rot] = rotNewBoard[idx];
                            rotNewMask[idx_rot] = rotNewMask[idx];
                        }
                    }
                }

                for (int i = 0; i < currBoard.length; i++) {
                    result[i] = rotNewBoard[i] * currMask[i]; // get previous board state
                    newTilesMask[i] = rotNewMask[i] ^ currMask[i]; // get mask of new tiles
                    newTiles[i] = rotNewBoard[i] * newTilesMask[i]; // get new tiles
                }

                if (Arrays.equals(result, currBoard)) {
                    break;
                } else if (rot == 3) {
                    return false;
                }
            }
        }

        for (int i = 0; i < 15; i++) {
            int start = i * 15;
            int end = start + 15;

            int[] row = Arrays.copyOfRange(rotNewBoard, start, end);
            Log.d(TAG, Arrays.toString(row));
        }

        List<String> newWords = extractWords();
        Log.d(TAG, String.valueOf(newWords));

        return true;
    }

    public List<int[]> extractWordIndices() {
        List<int[]> found = new ArrayList<>();
        int sz = 15;

        // HORIZONTAL SEQEUNCES
        for (int r = 0; r < sz; r++) {
            int runStart = -1;
            boolean sawNew = false;
            for (int c = 0; c <= sz; c++) {
                boolean occupied = (c < sz && rotNewBoard[r*sz + c] != 0);
                if (occupied) {
                    if (runStart < 0) {
                        runStart = c;
                        sawNew   = (newTilesMask[r*sz + c] == 1);
                    } else if (newTilesMask[r*sz + c] == 1) {
                        sawNew = true;
                    }
                }
                if (!occupied && runStart >= 0) {
                    int runLen = c - runStart;
                    if (runLen > 1 && sawNew) {
                        int[] cells = new int[runLen];
                        for (int j = 0; j < runLen; j++)
                            cells[j] = r*sz + (runStart + j);
                        found.add(cells);
                    }
                    runStart = -1;
                }
            }
        }

        // VERTICAL SEQUENCES
        for (int c = 0; c < sz; c++) {
            int runStart = -1;
            boolean sawNew = false;
            for (int r = 0; r <= sz; r++) {
                boolean occupied = (r < sz && rotNewBoard[r*sz + c] != 0);
                if (occupied) {
                    if (runStart < 0) {
                        runStart = r;
                        sawNew   = (newTilesMask[r*sz + c] == 1);
                    } else if (newTilesMask[r*sz + c] == 1) {
                        sawNew = true;
                    }
                }
                if (!occupied && runStart >= 0) {
                    int runLen = r - runStart;
                    if (runLen > 1 && sawNew) {
                        int[] cells = new int[runLen];
                        for (int i = 0; i < runLen; i++)
                            cells[i] = (runStart + i)*sz + c;
                        found.add(cells);
                    }
                    runStart = -1;
                }
            }
        }

        return found;
    }

    public List<String> extractWords() {
        int sz = 15;
        Set<String> found = new HashSet<>();

        for (int p = 0; p < newTilesMask.length; ++p) {
            if (newTilesMask[p] == 0) continue;

            int row = p / sz;
            int col = p % sz;

            // horizontal
            int r = col; // right
            while (r < sz-1 && rotNewBoard[row*sz+ r+1] != 0)
                r++;

            int l = col; // left
            while (l > 0 && rotNewBoard[row*sz+ l-1] != 0)
                l++;

            //Log.d(TAG, String.format("r:%d l:%d\n", r, l));

            if (r - l + 1 > 1)
                found.add(buildWord(row, l, 0, 1, r - l + 1, rotNewBoard));

            // vertical
            int t = row; // top
            while (t > 0 && rotNewBoard[(t-1)*sz + col] != 0)
                t++;

            int b = row; // bottom
            while (b < sz-1 && rotNewBoard[(b+1)*sz + col] != 0)
                b++;

            if (b - t + 1 > 1)
                found.add(buildWord(t, col, 1, 0, b - t + 1, rotNewBoard));
        }
        return new ArrayList<>(found);
    }

    private String buildWord(int row, int col,
                             int dr, int dc, int len,
                             int[] board) {
        StringBuilder w = new StringBuilder(len);
        for (int i = 0; i < len; ++i)
            w.append(CLASS_NAMES[board[(row + i*dr) * 15 + col + i*dc]]);
        return w.toString();
    }

    public boolean validateWord() {
        return true;
    }

    public LiveData<int[]> getBoardState() {
        return boardState;
    }

    public void setBoardState(int[] state) {
        boardState.postValue(state);
    }

    public LiveData<int[]> getNewBoardState() {
        return newBoardState;
    }

    public void setNewBoardState(int[] state) {
        newBoardState.postValue(state);
    }

    public void setPlayerCount(int newVal) {
        playerCount.setValue(newVal);
    }

    public Integer getPlayerCount() {
        return playerCount.getValue();
    }

    public int[] getPlayerScores() {
        return playerScores.getValue();
    }

    public void initPlayerScores()
    {
        int[] scores = new int[this.getPlayerCount()];
        for (int i=0; i < this.getPlayerCount(); i++) {
            scores[i] = 0;
        }
        playerScores.setValue(scores.clone());
    }

    public void setPlayerScore(int playerIndex, int newScore) {
        int[] scores = playerScores.getValue();

        if (scores == null ) {
            Log.d("GameStateViewModel.java", "PLAYER SCORES NULL");
            return;
        }

        scores[playerIndex] = newScore;
        playerScores.setValue(scores.clone()); // must trigger LiveData update
    }

    public int getPlayerScore(int playerIndex) {
        int[] scores = playerScores.getValue();
        assert scores != null;
        return scores[playerIndex];
    }

    public LiveData<Boolean> getIsBoardOriented() {
        return isBoardOriented;
    }

    public void setIsBoardOriented(boolean status) {
        isBoardOriented.postValue(status);
    }

    public Integer getTurnCount() {
        return turnCount.getValue();
    }

    public void resetTurnCount() {
        turnCount.setValue(0);
    }

    public void iterateTurnCount() {
        turnCount.setValue(getTurnCount() + 1);
    }

    public void setNewBoardMask(int[] state) {
        newBoardMask.postValue(state);
    }

    public LiveData<int[]> getNewBoardMask() {
        return newBoardMask;
    }

    public void setBoardMask(int[] state) {
        boardMask.postValue(state);
    }

    public LiveData<int[]> getBoardMask() {
        return boardMask;
    }

    private int[] safeCopy(LiveData<int[]> liveData, int size) {
        int[] value = liveData.getValue();
        return (value != null) ? Arrays.copyOf(value, value.length) : new int[size];
    }
}
