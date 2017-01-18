package util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import EntIndex.AhoCorasickDoubleArrayTrie;
import EntIndex.IndexBuilder;

public class MentionCounter {
	IndexBuilder ibd = null;
	HashMap<String, Integer> mentions = null;
	
	public List<Integer> findMention(List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> mentionList, int doc_lenth){
    	int[] sent = new int[doc_lenth];
    	List<Integer> re_mentions = new ArrayList<>();
    	for(int i=0;i<doc_lenth;i++)
    		sent[i] = -1;
    	AhoCorasickDoubleArrayTrie<String>.Hit<String> tmp_hit = null;
    	for(int i=0;i<mentionList.size();i++){
    		tmp_hit = mentionList.get(i);
    		int last_begin_index = sent[tmp_hit.begin];
    		int last_end_index = sent[tmp_hit.end-1];
    		//if both equal -1, then update; or equal other number, then do nothing
    		if(last_begin_index == -1 && last_end_index == -1){
    			for(int j=tmp_hit.begin;j<tmp_hit.end;j++)
    				sent[j] = i;
				continue;
    		}
    		else if(last_begin_index!=last_end_index){
    			int length_begin = last_begin_index==-1 ? 0:(mentionList.get(last_begin_index).end - mentionList.get(last_begin_index).begin);
    			int length_end = last_end_index==-1 ? 0:(mentionList.get(last_end_index).end - mentionList.get(last_end_index).begin);
    			int length_cur = tmp_hit.end - tmp_hit.begin;
    			if(length_cur>length_begin && length_cur>length_end){
    				for(int j=tmp_hit.begin;j<tmp_hit.end;j++)
        				sent[j] = i;
    				if(length_begin!=0)
    					for(int j=mentionList.get(last_begin_index).begin;j<tmp_hit.begin;j++)
            				sent[j] = -1;
    				if(length_end!=0)
    					for(int j=tmp_hit.end;j<mentionList.get(last_end_index).end;j++)
            				sent[j] = -1;
    			}
    		}
        }
    	int cur_index = -1;
    	for(int i=0;i<doc_lenth;i++){
    		if(sent[i]!=-1 && sent[i]!=cur_index){
    			re_mentions.add(sent[i]);
    			cur_index = sent[i];
    		}
    	}
    	return re_mentions;
    }
	
	public HashMap<String, Integer> countMention(String doc_path) throws Exception{
		BufferedReader fin = new BufferedReader(new FileReader(doc_path));
    	String line = null;
    	String label = null;
    	List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> can_mention_list = null;
    	Pattern p=Pattern.compile("\\[\\[(.*?)\\]\\]");
    	Matcher m=null;
    	List<Integer> mention_list = null;
    	HashMap<String, Integer> mention_count = new HashMap<>();
    	String tmp_str = null;
    	int line_count=0;
    	while((line = fin.readLine())!=null){
		line_count++;
		if(line_count%10000==0) System.out.printf("has processed %d lines.\n", line_count);
    		int start_pos = 0;
        	int end_pos = 0;
    		m = p.matcher(line);
    		while(m.find()){
    			end_pos = m.start();
    			tmp_str = line.substring(start_pos, end_pos);
    			can_mention_list = ibd.parseText(tmp_str);
    			mention_list = findMention(can_mention_list, end_pos-start_pos);
    			for(Integer index:mention_list){
    				label = tmp_str.substring(can_mention_list.get(index).begin, can_mention_list.get(index).end);
					if(mention_count.containsKey(label))
						mention_count.put(label, mention_count.get(label)+1);
					else
						mention_count.put(label, 1);
    			}
    			//add text mention count
    			start_pos = m.end();
    		}
    		end_pos = line.length();
    		if(start_pos!=end_pos){
    			tmp_str = line.substring(start_pos, end_pos);
    			can_mention_list = ibd.parseText(tmp_str);
    			mention_list = findMention(can_mention_list, end_pos-start_pos);
    			for(Integer index:mention_list){
    				label = tmp_str.substring(can_mention_list.get(index).begin, can_mention_list.get(index).end);

					if(mention_count.containsKey(label))
						mention_count.put(label, mention_count.get(label)+1);
					else
						mention_count.put(label, 1);
    			}
    		}
    		label = null;
    		can_mention_list = null;
    		mention_list = null;
    		m = null;
    		line = null;
    	}
    	fin.close();
    	return mention_count;
	}
	
	public void saveCount(String count_file) throws Exception{
	    BufferedWriter writer = new BufferedWriter(new FileWriter(count_file, false));
	    for (Map.Entry<String, Integer> entry : this.mentions.entrySet()) {
	    	writer.write(entry.getKey() + "::=" + entry.getValue()+"\n");
        } 
	    writer.close();
	}
	
	
	
	public static void main(String[] args) throws Exception{
		String count_file = null;
		MentionCounter mc = new MentionCounter();
		mc.ibd = new IndexBuilder();
		mc.ibd.initCons(args);
		mc.ibd.build();
		mc.ibd.save();
		//mc.ibd.load();
		mc.mentions = mc.countMention(mc.ibd.cons.doc_file);
		count_file = mc.ibd.cons.output_path+"mention_count";
	    mc.saveCount(count_file);
	}
}
