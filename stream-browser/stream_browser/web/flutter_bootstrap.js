// NOTE: BE CAREFUL SAVING FILE
// This file must be formatted correctly. With the VSCode setting 'format_on_save' it will format incorrectly and break
{{flutter_js}}
{{flutter_build_config}}

_flutter.loader.load({
    onEntrypointLoaded: async function(engineInitializer) {
        engineInitializer.initializeEngine(
        {
            useColorEmoji:true
        } 
        ).then(function(appRunner) {
          appRunner.runApp();
        });
  }});