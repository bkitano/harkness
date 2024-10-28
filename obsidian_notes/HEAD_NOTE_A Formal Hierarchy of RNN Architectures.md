# A Formal Hierarchy of RNN Architectures.md

The document titled "A Formal Hierarchy of RNN Architectures" presents a hierarchy to evaluate the expressive power of various recurrent neural network (RNN) architectures. The hierarchy is based on space complexity (memory requirements) and whether the recurrent update can be described by a weighted finite-state machine (rational recurrence). The document includes several theoretical proofs and experiments to group different RNNs—such as LSTM, QRNN, and GRU—based on these capabilities. 

Key findings indicate that LSTMs are not rationally recurrent (RR), meaning they have expressive power beyond weighted finite automata (WFAs). In contrast, models like the QRNN and GRU are RR but have limitations that can potentially be expanded by stacking or other methods. The hierarchy not only aids in understanding the state expressiveness of RNNs but also the languages they can recognize when paired with decoders. 

Experimental sections support the hierarchy's applicability to unsaturated RNNs, showing that the theoretical predictions align with practical results, such as the inability of single-layer QRNNs to recognize certain patterns without enhanced decoders. The results illuminate the varying capabilities of different RNN architectures and help inform future improvements and applications in natural language processing. 

In conclusion, this theoretical framework aids in bridging the understanding gap between complex RNN behaviors and their computational power, guiding the development of more efficient architectures. The document includes substantial mathematical background and results on WFAs and provides theoretical foundations for comparisons, presenting provable differences in RNN capacity and highlighting specific areas where non-rational models exceed others.

- [[RNN_Expressive_Capacity:_Space_Complexity_and_Rational_Recurrence]]
- [[State_and_Language_Expressiveness_in_RNNs]]
- [[Understanding_WFAs_and_Their_Relation_to_RNNs]]
- [[The_Role_of_Decoders_in_Language_Recognition_with_RNNs]]