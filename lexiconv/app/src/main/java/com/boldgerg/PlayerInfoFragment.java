package com.boldgerg;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.NumberPicker;

import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;

import java.util.ArrayList;
import java.util.List;

public class PlayerInfoFragment extends Fragment {
    private GameStateViewModel viewModel;

   @Override
    public View onCreateView(
    LayoutInflater inflater,
    ViewGroup container,
    Bundle savedInstanceState
    ) {
        return inflater.inflate(R.layout.fragment_playerinfo, container, false);
    }

    @Override
    public void onViewCreated(View view, Bundle savedInstanceState) {
        viewModel = new ViewModelProvider(requireActivity()).get(GameStateViewModel.class);

        NumberPicker numPick = view.findViewById(R.id.playerCountNumberPicker);
        LinearLayout inputList = view.findViewById(R.id.playerNameInputList);

        Button startBtn = view.findViewById(R.id.startGameButton);
        startBtn.setOnClickListener(v -> {
            int count = inputList.getChildCount();
            List<String> playerNames = new ArrayList<>();

            for (int i = 0; i < count; i++) {
                View child = inputList.getChildAt(i);
                if (child instanceof EditText) {
                    String name = ((EditText) child).getText().toString().trim();
                    if (name.isEmpty()) {
                        name = "Player " + (i + 1);
                    }
                    playerNames.add(name);
                }
            }

            viewModel.setPlayerNames(playerNames);
            viewModel.initNewGame();

            NavController nav = NavHostFragment.findNavController(PlayerInfoFragment.this);
            nav.navigate(R.id.action_playerInfoFragment_to_scoreboardFragment);
        });

        numPick.setMinValue(2);
        numPick.setMaxValue(4);
        numPick.setValue(2);
        viewModel.setPlayerCount(2);

        int[] currentCount = {2};

        for (int i = 0; i < currentCount[0]; i++) {
            EditText input = new EditText(requireContext());
            input.setHint("Player " + (i + 1));
            inputList.addView(input);
        }

        numPick.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker picker, int oldVal, int newVal) {
                viewModel.setPlayerCount(newVal);

                if (newVal > oldVal) {
                    for (int i = 0; i < newVal - oldVal; i++) {
                        EditText input = new EditText(requireContext());
                        input.setHint("Player " + (currentCount[0] + 1));
                        inputList.addView(input);
                        currentCount[0]++;
                    }
                } else {
                    for (int i = 0; i < oldVal - newVal; i++) {
                        int childIndex = inputList.getChildCount() - 1;
                        if (childIndex >= 0) {
                            inputList.removeViewAt(childIndex);
                            currentCount[0]--;
                        }
                    }
                }
                // Navigate to next fragment
                //NavController nav = NavHostFragment.findNavController();
                //nav.navigate(R.id.action_playerInfoFragment_to_boardParserFragment);
            }
        });
        //NumberPicker numPick = view.findViewById(R.id.playerCountNumberPicker);
   }
}
