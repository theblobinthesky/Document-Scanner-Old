package com.erikstern.documentscanner

import android.content.Context
import android.content.pm.PackageManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.erikstern.documentscanner.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val surfaceRenderer = GLSurfaceRenderer(this)

    private fun initSurfaceViewWithGivenPermissions() {
        surfaceRenderer.setSurfaceView(binding.glSurfaceView)
        binding.glSurfaceView.visibility = View.VISIBLE
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val permissionCheck = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grantResults ->
            val allResult = grantResults.all { it.value }
            if(allResult) initSurfaceViewWithGivenPermissions()
        }

        if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            initSurfaceViewWithGivenPermissions()
        }
        else {
            permissionCheck.launch(arrayOf(android.Manifest.permission.CAMERA))
        }
    }

    companion object {
        init {
            System.loadLibrary("documentscanner");
        }
    }
}