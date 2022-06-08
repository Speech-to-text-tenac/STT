// import {
//    ReactMediaRecorder,
//   //useReactMediaRecorder,
// } from "react-media-recorder";
// //import { ReactMediaRecorder, useReactMediaRecorder } from "react-media-recorder"
// import classes from "./Spectogram.module.css";
//import AudioReactRecorder, { RecordState } from 'audio-react-recorder'
import Plot from 'react-plotly.js'
export const Spectogram = () => {
// const {status, startRecording, stopRecording, mediaBlobUrl} = useReactMediaRecorder({video: true})
//  const handleSave = async (mediaBlobUrl) => {
//    const audioBlob = await fetch(mediaBlobUrl).then((r) => r.blob());
//    const audioFile = new File([audioBlob], "voice.wav", { type: "audio/wav" });
//    const formData = new FormData(); // preparing to send to the server

//    formData.append("file", audioFile); // preparing to send to the server
//    console.log(audioFile);
//    //  onSaveAudio(formData); // sending to the server
//  };
  return (
    <div>
      <h1>Bar plot</h1>
      <Plot data={[{
        x:[1,2,3], y:[1,2,3],
        type: 'bar',
        mode: 'lines+markers',
        maker: {color: 'red'}
      }]}
      layout={{width:500, height:300, title: 'simple bar plot'}}
      />
    </div>
    // <ReactMediaRecorder
    // audio
    // render={({ status, startRecording, stopRecording, mediaBlobUrl }) => (
    // <div className={classes.display}>
    //   <p>{status}</p>
    //   <button onClick={startRecording}>Start</button>
    //   <button onClick={stopRecording}>Stop</button>
    //   <button onClick={handleSave(mediaBlobUrl)}>Predict</button>
    //   <br></br>
    //   <br></br>
    //   <audio src={mediaBlobUrl} autoPlay loop controls></audio>
    //   <section>
    //     <p>HELLO, EVERYONE</p>
    //   </section>
    // </div>
    // )}
    // />
  ); 
}

