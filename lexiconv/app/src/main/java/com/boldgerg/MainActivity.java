package com.boldgerg;

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;
import androidx.navigation.ui.NavigationUI;

public class MainActivity extends AppCompatActivity {
    private NavController navController;

    private GameStateViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Fragment host = getSupportFragmentManager()
                .findFragmentById(R.id.nav_host_fragment);
        if (!(host instanceof NavHostFragment)) {
            throw new IllegalStateException(
                    "Activity must contain a NavHostFragment with id nav_host_fragment");
        }

        navController = ((NavHostFragment) host).getNavController();
    }

    @Override
    public boolean onSupportNavigateUp() {
        return navController.navigateUp() || super.onSupportNavigateUp();
    }
}
