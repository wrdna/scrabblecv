package com.boldgerg;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import androidx.fragment.app.Fragment;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;

public class HomeFragment extends Fragment {
    @Override
    public View onCreateView(
    LayoutInflater inflater,
    ViewGroup container,
    Bundle savedInstanceState
    ) {
        return inflater.inflate(R.layout.fragment_home, container, false);
    }

    @Override
    public void onViewCreated(View view, Bundle savedInstanceState) {
        Button newGameBtn = view.findViewById(R.id.newGameButton);
        newGameBtn.setOnClickListener(v -> {
            NavController nav = NavHostFragment.findNavController(this);
            nav.navigate(R.id.action_homeFragment_to_playerInfoFragment);
        });
    }
}
