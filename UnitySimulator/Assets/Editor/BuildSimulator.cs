using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class BuildSimulator
{
    [MenuItem("Tools/Aim and Shoot/Create scene")]
    public static void CreateScene() {
        Directory.CreateDirectory("Assets/Scenes");
        var scene=EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        var viewer=new GameObject("Synchronized dual Aim and Shoot viewer").AddComponent<DualAimShootViewer>();
        viewer.surfaceShader=Shader.Find("Standard");
        viewer.lineShader=Shader.Find("Sprites/Default");
        if(viewer.surfaceShader==null || viewer.lineShader==null)
            throw new System.Exception("Required built-in shaders are unavailable");
        EditorSceneManager.SaveScene(scene,"Assets/Scenes/AimAndShoot.unity");
        EditorBuildSettings.scenes=new[]{new EditorBuildSettingsScene("Assets/Scenes/AimAndShoot.unity",true)};
    }

    [MenuItem("Tools/Aim and Shoot/Build Windows app")]
    public static void Build() {
        CreateScene();
        PlayerSettings.companyName="Local Research";
        PlayerSettings.productName="Aim and Shoot Live AI";
        PlayerSettings.defaultScreenWidth=1600; PlayerSettings.defaultScreenHeight=900;
        PlayerSettings.fullScreenMode=FullScreenMode.Windowed;
        PlayerSettings.resizableWindow=true;
        PlayerSettings.runInBackground=true;
        PlayerSettings.SetScriptingBackend(UnityEditor.Build.NamedBuildTarget.Standalone,ScriptingImplementation.Mono2x);
        Directory.CreateDirectory("../UnityBuild");
        var report=BuildPipeline.BuildPlayer(new[]{"Assets/Scenes/AimAndShoot.unity"},
            "../UnityBuild/AimAndShoot.exe",BuildTarget.StandaloneWindows64,BuildOptions.None);
        if(report.summary.result!=UnityEditor.Build.Reporting.BuildResult.Succeeded)
            throw new System.Exception("Build failed: "+report.summary.result);
    }
}
