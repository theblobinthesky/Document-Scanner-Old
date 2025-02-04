package com.erikstern.documentscanner

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.res.AssetManager
import android.content.res.Configuration
import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.util.Log
import android.view.MotionEvent
import android.view.Surface
import android.view.Window
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class GLSurfaceRenderer(val activity: Activity) : GLSurfaceView.Renderer {
    lateinit var context: Context

    var surfaceTextureId = -1
    lateinit var surfaceTexture: SurfaceTexture
    lateinit var surface: Surface
    var surfaceChanged = false
    var nativeContext: Long = 0

    val lock = Any()
    var frameAvailable = false

    private external fun nativeDestroy()
    private external fun nativeInit(assetManager: AssetManager, surface: Surface, internalDataPath: String, preview_width: Int, preview_height: Int, enableDarkMode: Boolean)
    private external fun nativeMotionEvent(event: Int, x: Float, y: Float)
    private external fun nativeRender()

    @SuppressLint("ClickableViewAccessibility")
    fun setSurfaceView(surfaceView: GLSurfaceView) {
        Log.i("test", "setSurfaceView");

        surfaceView.setEGLContextClientVersion(3)
        surfaceView.setRenderer(this)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        context = surfaceView.context

        surfaceView.setOnTouchListener { _, motionEvent ->
            when(motionEvent.action) {
                MotionEvent.ACTION_DOWN, MotionEvent.ACTION_UP, MotionEvent.ACTION_MOVE -> {
                    Log.i("motion_event", motionEvent.action.toString())
                    nativeMotionEvent(motionEvent.action, motionEvent.x, motionEvent.y)
                }
            }
            true
        }
    }

    fun destroyRender() {
        nativeDestroy()
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        Log.i("test", "onSurfaceCreated");

        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        surfaceTextureId = textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, surfaceTextureId)

        surfaceTexture = SurfaceTexture(surfaceTextureId)
        surfaceTexture.setOnFrameAvailableListener {
            synchronized(lock) {
                frameAvailable = true
            }
        }

        surface = Surface(surfaceTexture)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        Log.i("test", "onSurfaceChanged");

        if(surfaceChanged) return
        surfaceChanged = true

        val enableDarkMode = when (context.resources?.configuration?.uiMode?.and(Configuration.UI_MODE_NIGHT_MASK)) {
            Configuration.UI_MODE_NIGHT_YES -> true
            else -> false
        }

        val internalDataPath = activity.filesDir.absolutePath
        nativeInit(context.assets, surface, internalDataPath, width, height, enableDarkMode)
        Log.d("docscanner", nativeContext.toString())
    }

    override fun onDrawFrame(gl: GL10?) {
        synchronized(lock) {
            if (frameAvailable) {
                surfaceTexture.updateTexImage()
                frameAvailable = false
            }
        }

        nativeRender()
    }
}