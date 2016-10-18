package Common;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
/**
 * Created by ethan on 16/3/23.
 */
public class Constant {
    public String label_list_path = null;
    public String output_path = null;
    public String input_path = null;
    public String trie_path = null;
    public String doc_file = null;
    public int max_index_num = 4000000;
    public int isMatchWords = 1;
    public String interval = "::=";
    
    public void parseArgs(String[] args){
    	if(args.length<2){
    		System.out.println("Accurate Label Matching toolkit v 0.1c\n\n");
    		System.out.println("Options:\n");
    		System.out.println("\t-label_file <file>\n");
    		System.out.println("\t\tUse label from <file> to build index for matching, each line in <file> is <lable>::=<value>\n");
    		System.out.println("\t-doc_file <file>\n");
    		System.out.println("\t\tCount the mentions in <file>\n");
    		System.out.println("\t-save_path <path>\n");
    		System.out.println("\t\tUse <path> to save index file\n");
    		System.out.println("\t-read_path <path>\n");
    		System.out.println("\t\tUse <path> to read index file\n");
    		System.out.println("\t-single_index_num <int>\n");
    		System.out.println("\t\tSingle index should less than <int>; default is 4000000;\n");
    		System.out.println("\t-is_match_words <int>\n");
    		System.out.println("\t\tif 1 only match words; default is 0;\n");
    	}
        for(int i=0;i<args.length;i++){
            if(args[i].equals("-label_file"))
            	label_list_path=args[i+1];
            if(args[i].equals("-doc_file"))
            	doc_file=args[i+1];
            if(args[i].equals("-save_path")){
            	output_path=args[i+1];
            	trie_path = output_path + "index_trie";
            }
            if(args[i].equals("-read_path")){
            	input_path=args[i+1];
            	trie_path = input_path + "index_trie";
            }
            if(args[i].equals("-single_index_num"))
            	max_index_num = Integer.parseInt(args[i+1]);
            if(args[i].equals("-is_match_words"))
            	isMatchWords = Integer.parseInt(args[i+1]);
            
        }
    }
}
