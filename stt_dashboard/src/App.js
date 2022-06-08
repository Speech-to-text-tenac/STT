import * as React from "react";
import { Recorder } from "react-voice-recorder";
import Records from "./elements/pages/record";
// import Home from "./view/pages/home"

function App() {
  return (
	  
    <div>
		<header></header>
			<Records />
		</div>
  );
}

  //   <Router>
  //     <navbar />
  //     <Switch>
  //       <Route path='/' exact component={Home} />
  //       <Route path='/record' component={Recorder} />
  //       <Route path='/Amharic' component={Amharic} />
  //       <Route path='/goats' component={Goats} />
  //     </Switch>
  //   </Router>
  // );


export default App;
