package com.boldgerg;

import android.content.Context;
import android.os.Bundle;
import android.text.InputFilter;
import android.text.InputType;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.NumberPicker;
import android.widget.TextView;

import androidx.annotation.ColorInt;
import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.Fragment;
import androidx.gridlayout.widget.GridLayout;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;
import androidx.recyclerview.widget.GridLayoutManager;

import com.google.android.material.color.utilities.Score;

import org.w3c.dom.Text;

public class ScoreboardFragment extends Fragment {
    private GameStateViewModel viewModel;

    @Override
    public View onCreateView(
    LayoutInflater inflater,
    ViewGroup container,
    Bundle savedInstanceState
    ) {
        return inflater.inflate(R.layout.fragment_scoreboard, container, false);
    }

    @Override
    public void onViewCreated(View view, Bundle savedInstanceState) {
        viewModel = new ViewModelProvider(requireActivity()).get(GameStateViewModel.class);

        LinearLayout playerScoresList = view.findViewById(R.id.playerScoresList);

        for (int i = 0; i < viewModel.getPlayerCount(); i++) {
            TextView score = new TextView(requireContext());
            String scoreText = getString(R.string.player_score,
                    viewModel.getPlayerNames().get(i),
                    viewModel.getPlayerScores()[i]);

            score.setText(scoreText);
            score.setTextSize(18);
            score.setPadding(8, 8, 8, 8);
            if (i == viewModel.getCurrentTurn())
            {
                score.setBackgroundResource(R.drawable.textview_border);
            }
            playerScoresList.addView(score);
        }

        GridLayout board = view.findViewById(R.id.scrabbleBoard);
        board.setEnabled(false);
        //board.setRowCount(15);
        //board.setColumnCount(15);
        //board.setUseDefaultMargins(false);
        //board.setPadding(0,0,0,0);

        //Context ctx = requireContext();
        //for (int row = 0; row < 15; row++) {
        //    for (int col = 0; col < 15; col++) {
        //        EditText cell = new EditText(ctx);

        //        GridLayout.LayoutParams lp = new GridLayout.LayoutParams(
        //                GridLayout.spec(row,1,1f),
        //                GridLayout.spec(col,1,1f)
        //        );
        //        lp.width = 0;  lp.height = 0;
        //        lp.setMargins(0,0,0,0);
        //        cell.setLayoutParams(lp);
        //        cell.setBackgroundResource(R.drawable.cell_border);

        //        cell.setFilters(new InputFilter[]{
        //                new InputFilter.LengthFilter(1)
        //        });
        //        cell.setInputType(
        //                InputType.TYPE_CLASS_TEXT |
        //                        InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS
        //        );

        //        cell.setGravity(Gravity.CENTER);
        //        cell.setIncludeFontPadding(false);
        //        cell.setPadding(0, 0, 0, 0);
        //        cell.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);

        //        //Character letter = viewModel.getBoardState().get(row * 15 + col);
        //        Character letter = null;
        //        cell.setText(letter == null ? "" : letter.toString());

        //        board.addView(cell);
        //    }
        //}
        Button scoreMoveCam = view.findViewById(R.id.scoreMoveCamera);
        scoreMoveCam.setOnClickListener(v -> {
            NavController nav = NavHostFragment.findNavController(ScoreboardFragment.this);
            nav.navigate(R.id.action_scoreboardFragment_to_boardParserFragment);
        });

        //NavController nav = NavHostFragment.findNavController();
        //nav.navigate(R.id.action_playerInfoFragment_to_boardParserFragment);
   }

}
