import React from "react";
import { Recorder } from "react-voice-recorder";
import "react-voice-recorder/dist/index.css";

export default class Records extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      audioURL: null,
      audioDetails: {
        url: null,
        blob: null,
        chunks: null,
        duration: {
          h: 0,
          m: 0,
          s: 0
        }
      }
    };
  }
  handleAudioStop(data) {
    console.log(data);
    this.setState({ audioDetails: data });
    //console.log(data);
  }
  handleAudioUpload(file) {
    console.log(file);
  }
  handleRest() {
    const reset = {
      url: null,
      blob: null,
      chunks: null,
      duration: {
        h: 0,
        m: 0,
        s: 0
      }
    };
    this.setState({ audioDetails: reset });
  }
  render() {
    return (
      <div className="App">
        <Recorder
          record={true}
          title={"አዲስ ድምፅ ለመቅረፅ..."}
          audioURL={this.state.audioDetails.url}
          showUIAudio
          handleAudioStop={(data) => this.handleAudioStop(data)}
          handleOnChange={(value) => this.handleOnChange(value, "firstname")}
          handleAudioUpload={(data) => this.handleAudioUpload(data)}
          handleRest={() => this.handleRest()}
        />
      </div>
    );
  }
}
